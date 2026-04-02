"""Streamlit entry point for the AgentsPype Inspector UI."""

from __future__ import annotations

import sys
from typing import Any

import streamlit as st

from .theme import inject_custom_css

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_module_paths() -> list[str]:
    """Parse module paths from CLI args passed after ``--``."""
    args = sys.argv[1:]
    if args:
        return args[0].split(",")
    return []


def _get_version() -> str:
    """Return the installed agentspype version, or '?' on failure."""
    try:
        from importlib.metadata import version

        return version("agentspype")
    except Exception:
        return "?"


def _instantiate(agent_class: type) -> Any:
    """Instantiate an agent with its default configuration."""
    from agentspype.agent.agent import Agent

    cls: type[Agent] = agent_class
    config = cls.definition.configuration_class()
    return cls(config)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_PAGE_ICONS = {
    "Overview": "\u25a6",  # ▦
    "Agents": "\u2b22",  # ⬢
    "Configuration": "\u2699",  # ⚙
}


def _render_sidebar(
    agent_classes: list[type], module_paths: list[str]
) -> tuple[str, type | None]:
    """Render the sidebar and return (page_name, selected_agent_class | None)."""
    with st.sidebar:
        st.markdown("### AgentsPype")
        st.markdown("Inspector")
        st.markdown(
            f'<div class="sidebar-version">v{_get_version()}</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigation",
            list(_PAGE_ICONS.keys()),
            format_func=lambda p: f"{_PAGE_ICONS[p]}  {p}",
            label_visibility="collapsed",
        )

        selected_class: type | None = None
        if page == "Agents":
            st.markdown(
                f'<div class="sidebar-section">Agents ({len(agent_classes)})</div>',
                unsafe_allow_html=True,
            )
            agent_names = [cls.__name__ for cls in agent_classes]
            selected_name = st.radio(
                "Select agent",
                agent_names,
                label_visibility="collapsed",
            )
            if selected_name:
                selected_class = next(
                    (c for c in agent_classes if c.__name__ == selected_name),
                    agent_classes[0],
                )

        with st.expander("Loaded modules"):
            for mp in module_paths:
                st.code(mp, language=None)

    return page, selected_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="AgentsPype Inspector",
        page_icon="\u2b22",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    module_paths = _get_module_paths()
    if not module_paths:
        st.error("No module paths provided. Run: `agentspype ui <module>`")
        return

    # Discover agent classes (cached in session state)
    if "agent_classes" not in st.session_state:
        with st.spinner("Discovering agents\u2026"):
            from agentspype.ui.discovery import discover_agent_classes

            st.session_state.agent_classes = discover_agent_classes(module_paths)

    agent_classes: list[type] = st.session_state.agent_classes

    if not agent_classes:
        st.error(f"No Agent subclasses found in: {', '.join(module_paths)}")
        return

    page, selected_class = _render_sidebar(agent_classes, module_paths)

    # Route to the selected page
    if page == "Overview":
        from .pages.overview import render_overview_page

        render_overview_page(agent_classes)

    elif page == "Agents":
        if selected_class is None:
            selected_class = agent_classes[0]
        from .pages.agents import render_agent_page

        render_agent_page(selected_class)

    elif page == "Configuration":
        from .pages.config import render_config_page

        render_config_page(agent_classes)


main()
