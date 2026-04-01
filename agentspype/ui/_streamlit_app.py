"""Streamlit entry point — loaded by `streamlit run`."""

from __future__ import annotations

import sys

import streamlit as st


def _get_module_paths() -> list[str]:
    """Parse module paths from CLI args passed after `--`."""
    args = sys.argv[1:]
    if args:
        return args[0].split(",")
    return []


def _render_agent_page(agent_class: type) -> None:
    from agentspype.ui.serializers import serialize_agent_class
    from agentspype.visualization.agent_visualization import AgentVisualization

    st.header(agent_class.__name__)
    st.caption(f"`{agent_class.__module__}`")

    data = serialize_agent_class(agent_class)

    # --- State machine diagram ---
    st.subheader("State Machine")
    try:
        viz = AgentVisualization()
        graph = viz.visualize(agent_class)
        st.graphviz_chart(graph.to_string())
    except Exception as exc:
        st.warning(f"Could not render diagram: {exc}")

    sm = data["state_machine"]
    if sm["states"]:
        st.subheader("States")
        st.dataframe(sm["states"], use_container_width=True, hide_index=True)

    if sm["transitions"]:
        st.subheader("Transitions")
        st.dataframe(sm["transitions"], use_container_width=True, hide_index=True)

    pub = data["publishing"]
    if pub["events"]:
        st.subheader("Publishing")
        st.dataframe(pub["events"], use_container_width=True, hide_index=True)

    listen = data["listening"]
    if listen["subscriptions"]:
        st.subheader("Listening")
        st.dataframe(listen["subscriptions"], use_container_width=True, hide_index=True)

    with st.expander("Configuration Schema"):
        st.json(data.get("configuration_schema", {}))

    with st.expander("Status Schema"):
        st.json(data.get("status_schema", {}))


def _render_overview_page(agent_classes: list[type]) -> None:
    from agentspype.ui.serializers import serialize_cross_agent_relationships
    from agentspype.visualization.cross_agent_visualization import (
        CrossAgentVisualization,
    )

    st.header("Cross-Agent Overview")

    try:
        viz = CrossAgentVisualization()
        graph = viz.visualize(agent_classes)
        st.graphviz_chart(graph.to_string())
    except Exception as exc:
        st.warning(f"Could not render overview diagram: {exc}")

    data = serialize_cross_agent_relationships(agent_classes)
    if data["event_wiring"]:
        st.subheader("Event Wiring")
        st.dataframe(data["event_wiring"], use_container_width=True, hide_index=True)


def _render_config_page(agent_classes: list[type]) -> None:
    import os
    import tempfile
    from pathlib import Path
    from typing import Any

    import yaml
    from pydantic import ValidationError
    from pydantic_wizard.streamlit_ui.model_form import render_model_form
    from pydantic_wizard.validation import validate_and_fix

    from agentspype.runner.config.wizard import (
        extract_config_fields,
        load_agentspype_configs,
        wrap_config_fields,
    )

    st.header("Configuration")

    agent_names = [cls.__name__ for cls in agent_classes]
    selected_name = st.selectbox("Agent", agent_names)
    if selected_name is None:
        return

    agent_class: Any = next(c for c in agent_classes if c.__name__ == selected_name)
    config_class = agent_class.definition.configuration_class

    tab_new, tab_validate = st.tabs(["New Config", "Validate Config"])

    with tab_new:
        st.subheader(f"New config for {selected_name}")
        with st.form("new_config_form"):
            data = render_model_form(config_class, key_prefix="new_")
            submitted = st.form_submit_button("Generate YAML")

        if submitted:
            instance = validate_and_fix(config_class, data)
            if instance is None:
                st.error("Validation failed.")
            else:
                fqn = f"{agent_class.__module__}.{agent_class.__name__}"
                module_path, _, class_name = fqn.rpartition(".")
                full = wrap_config_fields(
                    instance.model_dump(),
                    agent_class=class_name,
                    agent_path=module_path,
                )
                st.success("Valid configuration:")
                st.code(
                    yaml.dump(full, default_flow_style=False, allow_unicode=True),
                    language="yaml",
                )

    with tab_validate:
        st.subheader("Validate an existing YAML config")
        uploaded = st.file_uploader("Upload YAML", type=["yaml", "yml"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                configs = load_agentspype_configs(Path(tmp_path))
                for i, cfg in enumerate(configs):
                    _, _, fields = extract_config_fields(cfg)
                    try:
                        config_class.model_validate(fields)
                        st.success(f"Config {i + 1}: valid")
                    except ValidationError as e:
                        st.error(f"Config {i + 1}: invalid — {e}")
            finally:
                os.unlink(tmp_path)


def main() -> None:
    st.set_page_config(
        page_title="AgentsPype",
        page_icon="🤖",
        layout="wide",
    )

    module_paths = _get_module_paths()
    if not module_paths:
        st.error("No module paths provided. Run: agentspype ui <module>")
        return

    if "agent_classes" not in st.session_state:
        with st.spinner("Discovering agents..."):
            from agentspype.ui.discovery import discover_agent_classes

            st.session_state.agent_classes = discover_agent_classes(module_paths)

    agent_classes: list[type] = st.session_state.agent_classes

    if not agent_classes:
        st.error(f"No Agent subclasses found in: {', '.join(module_paths)}")
        return

    page: str = "Overview"
    with st.sidebar:
        st.title("🤖 AgentsPype")
        st.caption(f"{len(agent_classes)} agent(s) loaded")
        st.divider()

        page = st.radio(
            "View",
            ["Overview", "Agents", "Configuration"],
            label_visibility="collapsed",
        )

        if page == "Agents":
            st.divider()
            agent_names = [cls.__name__ for cls in agent_classes]
            selected = st.radio(
                "Select agent",
                agent_names,
                label_visibility="collapsed",
            )
            st.session_state.selected_agent = selected

    if page == "Overview":
        _render_overview_page(agent_classes)
    elif page == "Agents":
        selected_name = st.session_state.get(
            "selected_agent", agent_classes[0].__name__
        )
        agent_class = next(
            (c for c in agent_classes if c.__name__ == selected_name),
            agent_classes[0],
        )
        _render_agent_page(agent_class)
    elif page == "Configuration":
        _render_config_page(agent_classes)


main()
