"""Agent class discovery from module paths."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent

logger = logging.getLogger(__name__)


def discover_agent_classes(module_paths: list[str]) -> list[type[Agent]]:
    """Import modules and collect Agent subclasses (class-level only).

    Parameters
    ----------
    module_paths:
        Dotted module paths (e.g. ``["mypackage.agents.my_agent"]``).

    Returns
    -------
    list[type[Agent]]:
        Discovered Agent subclasses with a ``definition`` attribute,
        in order of discovery, deduplicated.
    """
    from agentspype.agent.agent import Agent

    discovered: list[type[Agent]] = []
    seen: set[type[Agent]] = set()

    for module_path in module_paths:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.exception("Failed to import module %s", module_path)
            continue
        except Exception:
            logger.exception("Unexpected error while importing module %s", module_path)
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Agent)
                and obj is not Agent
                and hasattr(obj, "definition")
                and obj not in seen
            ):
                seen.add(obj)
                discovered.append(obj)

    return discovered
