"""Agent detail page — per-agent metrics, diagram, state machine, events, schemas."""

from __future__ import annotations

from typing import Any

import streamlit as st

from agentspype.ui.serializers import serialize_agent_class

from ..components import (
    diagram_container,
    metric_card,
    render_events_table,
    render_metric_row,
    render_schema_properties,
    render_states_table,
    render_transitions_table,
    section_header,
)
from ..main import _instantiate
from ..metrics import compute_agent_metrics
from ..theme import NEUTRAL, PURPLE, SUCCESS, WARNING


def render_agent_page(agent_class: type) -> None:
    """Render the agent detail page."""
    data = serialize_agent_class(agent_class)
    metrics = compute_agent_metrics(data)

    section_header(
        agent_class.__name__,
        agent_class.__module__,
    )

    # --- Metric cards ---
    render_metric_row(
        [
            metric_card("States", metrics.state_count, SUCCESS),
            metric_card("Transitions", metrics.transition_count, WARNING),
            metric_card("Publishing", metrics.publishing_count, PURPLE),
            metric_card("Listening", metrics.listening_count, NEUTRAL),
        ]
    )

    # --- Tabs ---
    tab_diagram, tab_sm, tab_events, tab_schemas = st.tabs(
        [
            "Diagram",
            "State Machine",
            "Events",
            "Schemas",
        ]
    )

    with tab_diagram:
        _render_diagram_tab(agent_class)

    with tab_sm:
        _render_state_machine_tab(agent_class, data)

    with tab_events:
        _render_events_tab(data)

    with tab_schemas:
        _render_schemas_tab(data)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_diagram_tab(agent_class: type) -> None:
    """Full comprehensive agent diagram."""
    try:
        from agentspype.visualization.agent_visualization import AgentVisualization

        instance = _instantiate(agent_class)
        viz = AgentVisualization()
        graph = viz.visualize(instance)
        diagram_container(graph.to_string(), label="Comprehensive Agent Diagram")
    except Exception as exc:
        st.warning(f"Could not render agent diagram: {exc}")


def _render_state_machine_tab(agent_class: type, data: dict[str, Any]) -> None:
    """State machine diagram + states/transitions tables side by side."""
    sm = data["state_machine"]

    col_diagram, col_data = st.columns([3, 2])

    with col_diagram:
        try:
            from agentspype.visualization.state_machine_visualization import (
                StateMachineVisualization,
            )

            instance = _instantiate(agent_class)
            viz = StateMachineVisualization()
            graph = viz.create_visualization(
                instance.machine,
                current_state=instance.machine.current_state,
            )
            diagram_container(graph.to_string(), label="State Machine")
        except Exception as exc:
            st.warning(f"Could not render state machine: {exc}")

    with col_data:
        st.markdown("**States**")
        render_states_table(sm.get("states", []))
        st.markdown("")
        st.markdown("**Transitions**")
        render_transitions_table(sm.get("transitions", []))


def _render_events_tab(data: dict[str, Any]) -> None:
    """Publishing and listening tables side by side."""
    col_pub, col_listen = st.columns(2)

    with col_pub:
        st.markdown(f"**Publishing** ({len(data['publishing'].get('events', []))})")
        render_events_table(data["publishing"].get("events", []), kind="publishing")

    with col_listen:
        st.markdown(
            f"**Listening** ({len(data['listening'].get('subscriptions', []))})"
        )
        render_events_table(
            data["listening"].get("subscriptions", []), kind="listening"
        )


def _render_schemas_tab(data: dict[str, Any]) -> None:
    """Configuration and status schemas side by side."""
    config_schema = data.get("configuration_schema", {})
    status_schema = data.get("status_schema", {})

    col_config, col_status = st.columns(2)

    with col_config:
        st.markdown("**Configuration Schema**")
        if config_schema.get("properties"):
            render_schema_properties(config_schema, title="Configuration")
        else:
            st.info("Default configuration (no custom fields)")
        with st.expander("Raw JSON Schema"):
            st.json(config_schema)

    with col_status:
        st.markdown("**Status Schema**")
        if status_schema.get("properties"):
            render_schema_properties(status_schema, title="Status")
        else:
            st.info("Default status (no custom fields)")
        with st.expander("Raw JSON Schema"):
            st.json(status_schema)
