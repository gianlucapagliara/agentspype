"""Overview page — system-wide metrics and cross-agent diagram."""

from __future__ import annotations

import streamlit as st

from agentspype.ui.serializers import (
    serialize_agent_class,
    serialize_cross_agent_relationships,
)

from ..components import (
    diagram_container,
    metric_card,
    render_metric_row,
    render_wiring_table,
    section_header,
)
from ..main import _instantiate
from ..metrics import compute_overview_metrics
from ..theme import DANGER, NEUTRAL, PRIMARY, PURPLE, SUCCESS, WARNING


def render_overview_page(agent_classes: list[type]) -> None:
    """Render the system overview page."""
    section_header("System Overview", f"{len(agent_classes)} agent(s) discovered")

    # Serialize all agents and wiring
    all_data = [serialize_agent_class(cls) for cls in agent_classes]
    wiring_data = serialize_cross_agent_relationships(agent_classes)
    metrics = compute_overview_metrics(all_data, wiring_data)

    # --- Metric cards ---
    render_metric_row(
        [
            metric_card("Agents", metrics.agent_count, PRIMARY),
            metric_card("States", metrics.total_states, SUCCESS),
            metric_card("Transitions", metrics.total_transitions, WARNING),
            metric_card("Publishing", metrics.total_publishing, PURPLE),
            metric_card("Listening", metrics.total_listening, NEUTRAL),
            metric_card("Wiring", metrics.wiring_count, DANGER),
        ]
    )

    # --- Cross-agent diagram ---
    st.markdown("")  # spacer
    try:
        from agentspype.visualization.cross_agent_visualization import (
            CrossAgentVisualization,
        )

        instances = [_instantiate(cls) for cls in agent_classes]
        viz = CrossAgentVisualization()
        graph = viz.visualize(instances)
        diagram_container(graph.to_string(), label="Agent Network")
    except Exception as exc:
        st.warning(f"Could not render overview diagram: {exc}")

    # --- Event wiring table ---
    wiring = wiring_data.get("event_wiring", [])
    if wiring:
        st.markdown("")
        section_header("Event Wiring")
        render_wiring_table(wiring)
