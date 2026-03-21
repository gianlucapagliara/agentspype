"""Agent scaffolding utility -- create a new agent from the template."""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _to_snake_case(name: str) -> str:
    """Convert a CamelCase name to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _to_camel_case(name: str) -> str:
    """Convert a string to CamelCase.

    Handles both snake_case input (``my_agent``) and already-CamelCase
    input (``MyAgent``).
    """
    if "_" in name:
        return "".join(part.capitalize() for part in name.split("_"))
    # Already CamelCase (or single word) -- ensure first letter is upper
    return name[0].upper() + name[1:] if name else name


_runner_dir = os.path.dirname(os.path.abspath(__file__))
_package_dir = os.path.dirname(_runner_dir)
TEMPLATE_FOLDER = os.path.join(_package_dir, "template")


def copy_template(agent_name: str, output_dir: Path | str) -> Path:
    """Copy the agent template to *output_dir*, renaming ``Template`` to *agent_name*.

    Parameters
    ----------
    agent_name:
        The agent name.  Can be CamelCase (``MyAgent``) or snake_case
        (``my_agent``).  Class names will use the CamelCase form.
        If the name ends with "Agent" (e.g. ``MyFancyAgent``), the suffix
        is stripped for the prefix replacement so that ``TemplateAgent``
        becomes ``MyFancyAgent`` (not ``MyFancyAgentAgent``).
    output_dir:
        Destination directory.  Created if it does not exist.

    Returns
    -------
    Path
        The resolved *output_dir* where the files were written.
    """
    output_path = Path(output_dir)
    camel_name = _to_camel_case(agent_name)

    # Strip trailing "Agent" so TemplateAgent -> {prefix}Agent, not {prefix}AgentAgent
    if camel_name.endswith("Agent") and camel_name != "Agent":
        camel_name = camel_name[: -len("Agent")]

    # Build a dotted import path from the output directory (relative parts only)
    parts = [p for p in output_path.parts if p != "/"]
    import_path = ".".join(parts)

    logger.info(
        "Creating agent '%s' in '%s' with import path '%s'",
        camel_name,
        output_path,
        import_path,
    )

    if output_path.exists():
        logger.info(
            "Directory '%s' already exists: only missing files will be copied.",
            output_path,
        )
    output_path.mkdir(parents=True, exist_ok=True)

    logger.debug("Copying files from %s to %s", TEMPLATE_FOLDER, output_path)
    for filename in os.listdir(TEMPLATE_FOLDER):
        template_file = os.path.join(TEMPLATE_FOLDER, filename)
        dest_file = output_path / filename
        if filename.endswith(".py") and not dest_file.exists():
            logger.debug("Copying %s to %s...", filename, output_path)
            shutil.copy2(template_file, dest_file)

            # Replace template references
            content = dest_file.read_text()
            content = content.replace("agentspype.template.", f"{import_path}.")
            content = content.replace("Template", camel_name)
            dest_file.write_text(content)

    logger.info("Agent '%s' created successfully!", camel_name)
    return output_path
