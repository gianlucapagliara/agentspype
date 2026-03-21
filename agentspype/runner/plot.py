"""Agent diagram generation utility."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_agents(
    agent_module_paths: list[str],
    output_dir: Path | str = ".diagrams",
) -> None:
    """Import agent modules and call :meth:`visualize` on each agent class.

    Parameters
    ----------
    agent_module_paths:
        Dotted module paths (e.g. ``["mypackage.agents.my_agent"]``).
        Each module is expected to contain an :class:`Agent` subclass.
    output_dir:
        Directory where diagram files are saved.
    """
    from agentspype.agent.agent import Agent

    output = str(output_dir)

    for module_path in agent_module_paths:
        mod = importlib.import_module(module_path)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Agent)
                and obj is not Agent
                and hasattr(obj, "definition")
            ):
                try:
                    # Instantiate with a default config to generate diagram
                    config = obj.definition.configuration_class()
                    instance = obj(config)
                    instance.visualize(
                        save_file=True,
                        output_dir=output,
                    )
                    logger.info("Diagram generated for %s", obj.__name__)
                except Exception:
                    logger.exception("Error producing diagram for %s", obj.__name__)
