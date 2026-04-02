"""Pure data-computation helpers for dashboard metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentMetrics:
    """Summary counts for a single agent."""

    state_count: int
    transition_count: int
    publishing_count: int
    listening_count: int


@dataclass(frozen=True)
class OverviewMetrics:
    """Aggregate counts across all agents."""

    agent_count: int
    total_states: int
    total_transitions: int
    total_publishing: int
    total_listening: int
    wiring_count: int


def compute_agent_metrics(agent_data: dict[str, Any]) -> AgentMetrics:
    """Derive summary metrics from a serialized agent dict."""
    sm = agent_data.get("state_machine", {})
    pub = agent_data.get("publishing", {})
    listen = agent_data.get("listening", {})
    return AgentMetrics(
        state_count=len(sm.get("states", [])),
        transition_count=len(sm.get("transitions", [])),
        publishing_count=len(pub.get("events", [])),
        listening_count=len(listen.get("subscriptions", [])),
    )


def compute_overview_metrics(
    all_agent_data: list[dict[str, Any]],
    wiring_data: dict[str, Any],
) -> OverviewMetrics:
    """Derive aggregate metrics from all serialized agents + wiring."""
    total_s = total_t = total_p = total_l = 0
    for d in all_agent_data:
        m = compute_agent_metrics(d)
        total_s += m.state_count
        total_t += m.transition_count
        total_p += m.publishing_count
        total_l += m.listening_count

    return OverviewMetrics(
        agent_count=len(all_agent_data),
        total_states=total_s,
        total_transitions=total_t,
        total_publishing=total_p,
        total_listening=total_l,
        wiring_count=len(wiring_data.get("event_wiring", [])),
    )
