"""Interactive configuration wizard for agentspype agents.

Bridges ``pydantic-wizard`` and agentspype's YAML config format so that users
can interactively create, edit, validate, and inspect agent configurations.

Requires the ``wizard`` extra: ``pip install agentspype[wizard]``

All ``pydantic-wizard`` imports are lazy (inside function bodies) because the
package is an optional dependency.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent
    from agentspype.agent.configuration import AgentConfiguration


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_agent_fqn(
    agent_fqn: str,
) -> tuple[type[Agent], type[AgentConfiguration]]:
    """Resolve a fully-qualified agent class name to *(AgentClass, ConfigClass)*.

    Parameters
    ----------
    agent_fqn:
        Dotted path such as ``mypackage.agents.my_agent.MyAgent``.

    Returns
    -------
    tuple
        The resolved :class:`Agent` subclass and its ``configuration_class``
        obtained from ``Agent.definition``.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class does not exist in the module.
    """
    module_path, _, class_name = agent_fqn.rpartition(".")
    if not module_path:
        raise ValueError(f"agent_fqn must be a dotted path (got {agent_fqn!r})")
    module = import_module(module_path)
    agent_class: type[Agent] = getattr(module, class_name)
    return agent_class, agent_class.definition.configuration_class


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

#: Routing keys used in the current flat config format.
ROUTING_KEYS: frozenset[str] = frozenset(("agent_class", "agent_path"))

#: All routing keys (current + legacy) that must be stripped from config
#: fields before passing data to Pydantic models.
ALL_ROUTING_KEYS: frozenset[str] = frozenset(
    (
        "agent_class",
        "agent_path",
        "agent_module_path",
        "agent_class_name",
        "agent_name",
        "agent_configuration",
    )
)


def extract_config_fields(
    agent_config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Split an agentspype config dict into routing info and model fields.

    Supports both the new format (``agent_class`` / ``agent_path``) and the
    legacy format (``agent_class_name`` / ``agent_module_path``).

    Returns
    -------
    tuple
        ``(agent_class, agent_path, config_fields_dict)`` where
        *agent_class* is the class name and *agent_path* is the dotted
        module path (may be empty).
    """
    # New format takes precedence; fall back to legacy keys.
    class_name: str = (
        agent_config.get("agent_class", "")
        or agent_config.get("agent_class_name", "")
        or agent_config.get("agent_name", "")
    )
    agent_path: str = agent_config.get("agent_path", "") or agent_config.get(
        "agent_module_path", ""
    )

    fields = {k: v for k, v in agent_config.items() if k not in ALL_ROUTING_KEYS}
    return class_name, agent_path, fields


def wrap_config_fields(
    data: dict[str, Any],
    agent_class: str,
    agent_path: str = "",
) -> dict[str, Any]:
    """Wrap config fields back into agentspype flat format with routing keys."""
    result: dict[str, Any] = {"agent_class": agent_class}
    if agent_path:
        result["agent_path"] = agent_path
    result.update(data)
    return result


# ---------------------------------------------------------------------------
# YAML I/O (agentspype flat format)
# ---------------------------------------------------------------------------


def load_agentspype_configs(path: Path) -> list[dict[str, Any]]:
    """Load an agentspype YAML config file.

    Handles:
    - YAML list at root (multiple agents)
    - Legacy ``configuration_data`` envelope format
    - Bare single-agent dict (flat format)

    Returns
    -------
    list[dict[str, Any]]
        One dict per agent configuration found in the file.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # YAML list at root (new multi-agent format).
    if isinstance(raw, list):
        return list(raw)

    # Legacy envelope format.
    if "configuration_data" in raw:
        configs = raw["configuration_data"]
        if isinstance(configs, dict):
            configs = [configs]
        return list(configs)

    # Single agent config (flat dict).
    if isinstance(raw, dict):
        return [raw]

    return []


def save_agentspype_configs(configs: list[dict[str, Any]], path: Path) -> None:
    """Save agent configs to an agentspype YAML file.

    Writes a YAML list at root for multiple agents; a flat dict for a single
    agent.  Leverages ``pydantic-wizard``'s :class:`ModelConfigDumper` for
    rich type serialization.
    """
    from pydantic_wizard.serialization import (
        ModelConfigDumper,
        prepare_for_serialization,
    )

    prepared = [prepare_for_serialization(c) for c in configs]
    document: Any = prepared if len(prepared) != 1 else prepared[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            document,
            f,
            Dumper=ModelConfigDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


# ---------------------------------------------------------------------------
# Interactive wizard flows
# ---------------------------------------------------------------------------


def wizard_new(
    agent_fqn: str,
    output: Path,
    *,
    append: bool = False,
) -> None:
    """Interactively create a new agent configuration.

    Parameters
    ----------
    agent_fqn:
        Dotted path to the agent class
        (e.g. ``mypackage.agents.my_agent.MyAgent``).
    output:
        Destination YAML file.
    append:
        When *True* and *output* already exists, the new configuration is
        appended to the existing file instead of overwriting it.
    """
    from pydantic_wizard import display_summary_table, prompt_model
    from pydantic_wizard.display import display_error, display_success
    from pydantic_wizard.validation import validate_and_fix

    agent_class, config_class = resolve_agent_fqn(agent_fqn)

    # Derive routing keys from the FQN (dotted module path).
    module_path, _, class_name = agent_fqn.rpartition(".")

    # Prompt the user for every config field.
    data = prompt_model(config_class)

    # Validate (with interactive repair on failure).
    instance = validate_and_fix(config_class, data)
    if instance is None:
        display_error("Configuration aborted.")
        return

    # Use the validated data from the model instance.
    validated_data = instance.model_dump()

    display_summary_table(validated_data, model_name=class_name)

    full_config = wrap_config_fields(
        validated_data,
        agent_class=class_name,
        agent_path=module_path,
    )

    if append and output.exists():
        existing = load_agentspype_configs(output)
        existing.append(full_config)
        save_agentspype_configs(existing, output)
    else:
        save_agentspype_configs([full_config], output)

    display_success(f"Configuration saved to {output}")


def wizard_edit(
    config_path: Path,
    output: Path | None = None,
) -> None:
    """Interactively edit an existing agent configuration.

    If the file contains multiple agents the user is prompted to select which
    one to edit.

    Parameters
    ----------
    config_path:
        Path to the existing agentspype YAML config.
    output:
        Optional alternative destination.  Defaults to overwriting
        *config_path*.
    """
    import questionary
    from pydantic_wizard import display_summary_table, prompt_model
    from pydantic_wizard.display import display_error, display_success
    from pydantic_wizard.validation import validate_and_fix

    from agentspype.runner.config.loader import resolve_agent_from_config

    configs = load_agentspype_configs(config_path)
    if not configs:
        display_error(f"No agent configurations found in {config_path}")
        return

    # Select agent to edit when the file contains several.
    if len(configs) > 1:
        choices = [
            f"{c.get('agent_class', c.get('agent_class_name', '?'))} "
            f"({c.get('agent_path', c.get('agent_module_path', '?'))})"
            for c in configs
        ]
        selected = questionary.select("Select agent to edit:", choices=choices).ask()
        if selected is None:
            return
        idx = choices.index(selected)
    else:
        idx = 0

    config = configs[idx]
    class_name, agent_path, fields = extract_config_fields(config)

    agent_cls, config_class = resolve_agent_from_config(config)

    # Re-prompt with current values as defaults.
    data = prompt_model(config_class, defaults=fields)

    instance = validate_and_fix(config_class, data)
    if instance is None:
        display_error("Edit aborted.")
        return

    validated_data = instance.model_dump()
    display_summary_table(validated_data, model_name=class_name)

    configs[idx] = wrap_config_fields(
        validated_data,
        agent_class=class_name,
        agent_path=agent_path,
    )

    target = output or config_path
    save_agentspype_configs(configs, target)
    display_success(f"Configuration saved to {target}")


def wizard_validate(
    config_path: Path,
    *,
    interactive: bool = False,
) -> bool:
    """Validate all agent configurations in *config_path*.

    Returns
    -------
    bool
        *True* when every configuration entry is valid (or was successfully
        repaired in interactive mode).

    Parameters
    ----------
    config_path:
        Path to the agentspype YAML config file.
    interactive:
        When *True* and ``stdin`` is a TTY, invalid entries trigger an
        interactive repair flow.  Fixed configs are written back to
        *config_path*.
    """
    from pydantic import ValidationError
    from pydantic_wizard.display import (
        display_error,
        display_success,
        display_validation_errors,
    )

    from agentspype.runner.config.loader import resolve_agent_from_config

    configs = load_agentspype_configs(config_path)
    if not configs:
        display_error(f"No agent configurations found in {config_path}")
        return False

    all_valid = True
    modified = False

    for i, config in enumerate(configs):
        class_name, agent_path, fields = extract_config_fields(config)
        label = f"[{i + 1}/{len(configs)}] {class_name}"

        try:
            agent_cls, config_class = resolve_agent_from_config(config)
        except (ImportError, AttributeError, ValueError) as exc:
            display_error(f"{label}: cannot resolve agent class \u2014 {exc}")
            all_valid = False
            continue

        try:
            config_class.model_validate(fields)
            display_success(f"{label}: valid")
        except ValidationError as exc:
            display_error(f"{label}: invalid")
            display_validation_errors(exc.errors())
            all_valid = False

            if interactive and sys.stdin.isatty():
                from pydantic_wizard.validation import validate_and_fix

                instance = validate_and_fix(config_class, fields)
                if instance is not None:
                    configs[i] = wrap_config_fields(
                        instance.model_dump(),
                        agent_class=class_name,
                        agent_path=agent_path,
                    )
                    modified = True

    if modified:
        save_agentspype_configs(configs, config_path)
        from pydantic_wizard.display import display_success as _ds

        _ds(f"Fixed configurations saved to {config_path}")

    return all_valid or modified


def wizard_show(agent_fqn: str) -> None:
    """Display the configuration schema for an agent.

    Parameters
    ----------
    agent_fqn:
        Dotted path to the agent class.
    """
    from pydantic_wizard import display_schema, introspect_model

    _, config_class = resolve_agent_fqn(agent_fqn)
    specs = introspect_model(config_class)
    display_schema(config_class, specs)
