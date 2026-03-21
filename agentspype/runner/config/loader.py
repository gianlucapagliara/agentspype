"""Agent YAML config parsing and agent class resolution utilities."""

from __future__ import annotations

import hashlib
import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any

from agentspype.agent.agent import Agent


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


def load_agent_configs(
    config: dict[str, Any],
    instance_prefix: str,
) -> list[dict[str, Any]]:
    """Parse a pre-loaded agent config dict.

    Applies ``$INSTANCE_PREFIX`` substitution, unwraps the optional
    ``configuration_data`` envelope, and returns a list of raw agent
    configuration dicts.
    """
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
