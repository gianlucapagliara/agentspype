"""Tests for agentspype.ui.app.launch()."""

from __future__ import annotations

import builtins
import sys

import pytest


def test_launch_raises_import_error_without_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """launch() raises ImportError when streamlit is not installed."""
    # Remove streamlit from sys.modules to force re-import
    streamlit_modules = {
        k: v for k, v in sys.modules.items() if k.startswith("streamlit")
    }
    for key in streamlit_modules:
        monkeypatch.delitem(sys.modules, key)

    real_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "streamlit":
            raise ImportError("No module named 'streamlit'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Force reimport of the module too
    if "agentspype.ui.app" in sys.modules:
        monkeypatch.delitem(sys.modules, "agentspype.ui.app")

    from agentspype.ui.app import launch

    with pytest.raises(ImportError, match="agentspype\\[ui\\]"):
        launch(["some.module"])
