"""Agent YAML config parsing and agent class resolution utilities."""

from __future__ import annotations

import hashlib
import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration

ROUTING_KEYS: frozenset[str] = frozenset(("agent_class", "agent_path"))

LEGACY_ROUTING_KEYS: frozenset[str] = frozenset(
    (
        "agent_module_path",
        "agent_class_name",
        "agent_name",
        "agent_path",
        "agent_configuration",
    )
)


def resolve_agent_class(agent_module_path: str, agent_class_name: str) -> type[Agent]:
    """Resolve an agent class from its module path and class name.

    ``agent_module_path`` uses slash notation (e.g.
    ``mypackage/agents/my_agent/agent``), which is converted to a dotted
    import path before importing.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class does not exist in the module.
    """
    import_path = agent_module_path.replace("/", ".").rstrip(".")
    try:
        module = import_module(import_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import agent module '{import_path}' "
            f"(from path '{agent_module_path}'): {exc}"
        ) from exc
    try:
        agent_class: type[Agent] = getattr(module, agent_class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{import_path}' has no class '{agent_class_name}'"
        ) from exc
    return agent_class


def generate_instance_prefix(config_file: Path) -> str:
    """Generate a short instance prefix from config file name + timestamp.

    Returns the first 10 hex characters of the MD5 hash of
    ``<stem>_<unix_timestamp>``.
    """
    raw = f"{config_file.stem}_{int(time.time())}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def _apply_replacements(
    config: dict[str, Any],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Apply string substitutions to a config dict via JSON round-trip."""
    raw = json.dumps(config)
    for old, new in pairs:
        raw = raw.replace(old, new)
    result: dict[str, Any] = json.loads(raw)
    return result


def resolve_agent_from_config(
    raw: dict[str, Any],
) -> tuple[type[Agent], type[AgentConfiguration]]:
    """Resolve agent and configuration classes from a raw config dict.

    Supports:
    - New format: ``agent_class`` (+ optional ``agent_path``)
    - Legacy format: ``agent_module_path`` + ``agent_class_name``
    - bl_agents format: ``agent_name`` + ``agent_path``

    Raises
    ------
    ValueError
        If the config dict does not contain enough information to resolve
        the agent class.
    """
    from agentspype.agency import Agency

    # Determine the class name and optional module path from the config.
    agent_class_name: str | None = (
        raw.get("agent_class") or raw.get("agent_class_name") or raw.get("agent_name")
    )
    agent_path: str | None = raw.get("agent_path") or raw.get("agent_module_path")

    if agent_class_name is None:
        raise ValueError(
            "Config dict must contain 'agent_class', 'agent_class_name', "
            "or 'agent_name'"
        )

    # Try resolving via the Agency registry first.
    try:
        agent_cls = Agency.resolve_by_name(agent_class_name)
        config_cls = Agency._agent_to_configuration[agent_cls]
        return agent_cls, config_cls
    except (KeyError, ValueError):
        pass

    # Fall back to module-path-based resolution.
    if agent_path is None:
        raise ValueError(
            f"Agent class '{agent_class_name}' is not registered and no "
            "'agent_path' / 'agent_module_path' was provided for import"
        )

    agent_cls = resolve_agent_class(agent_path, agent_class_name)
    config_cls = agent_cls.definition.configuration_class
    return agent_cls, config_cls


def extract_config_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Strip routing keys and return flat configuration fields.

    If the legacy ``agent_configuration`` key is present its contents are
    merged into the returned dict.
    """
    all_routing_keys = ROUTING_KEYS | LEGACY_ROUTING_KEYS
    nested: dict[str, Any] = raw.get("agent_configuration", {}) or {}

    result: dict[str, Any] = {k: v for k, v in raw.items() if k not in all_routing_keys}
    result.update(nested)
    return result


def load_agent_configs(
    config: dict[str, Any] | list[dict[str, Any]],
    instance_prefix: str,
) -> list[dict[str, Any]]:
    """Parse a pre-loaded agent config dict or list.

    Applies ``$INSTANCE_PREFIX`` substitution, unwraps the optional
    ``configuration_data`` envelope, and returns a list of raw agent
    configuration dicts.  Accepts both a single dict and a list of dicts
    at the top level.
    """
    # Normalise list-at-root into a wrapper dict for the replacement step.
    if isinstance(config, list):
        wrapper: dict[str, Any] = {"configuration_data": config}
        replaced_wrapper = _apply_replacements(
            wrapper,
            [
                ("$INSTANCE_PREFIX", instance_prefix),
                ('"None"', "null"),
            ],
        )
        result: list[dict[str, Any]] = replaced_wrapper["configuration_data"]
        return result

    replaced_config = _apply_replacements(
        config,
        [
            ("$INSTANCE_PREFIX", instance_prefix),
            ('"None"', "null"),
        ],
    )

    controller_config: list[dict[str, Any]] | dict[str, Any] = replaced_config

    if "configuration_data" in replaced_config:
        controller_config = replaced_config["configuration_data"]

    if isinstance(controller_config, dict):
        controller_config = [controller_config]

    return controller_config
