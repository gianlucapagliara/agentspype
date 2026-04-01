"""Streamlit application for agentspype — agent inspection and config management."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch(
    module_paths: list[str],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Launch the Streamlit UI for the given agent modules.

    Parameters
    ----------
    module_paths:
        Dotted module paths containing Agent subclasses.
    host:
        Host to bind the Streamlit server.
    port:
        Port number.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Streamlit is required for the UI. Install with: pip install agentspype[ui]"
        ) from e

    app_path = Path(__file__).parent / "_streamlit_app.py"
    modules_arg = ",".join(module_paths)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            f"--server.address={host}",
            f"--server.port={port}",
            "--server.headless=true",
            "--",
            modules_arg,
        ],
        check=False,
    )
