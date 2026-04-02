"""Reusable UI atoms for the AgentsPype Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------


def metric_card(
    label: str, value: int | str, color: str, sub: str | None = None
) -> str:
    """Return the HTML for a single metric card.

    The caller is responsible for wrapping multiple cards in a
    ``<div class="metric-row">`` and rendering via ``st.markdown``.
    """
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card" style="border-left-color:{color}">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value}</div>'
        f"  {sub_html}"
        f"</div>"
    )


def render_metric_row(cards: list[str]) -> None:
    """Render a horizontal row of metric card HTML strings."""
    inner = "\n".join(cards)
    st.markdown(
        f'<div class="metric-row">{inner}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def badge(text: str, variant: str = "normal") -> str:
    """Return an HTML ``<span>`` badge pill.

    Variants: initial, final, normal, event, internal, primary, teal.
    """
    return f'<span class="badge badge-{variant}">{text}</span>'


def state_type_badge(state: dict[str, Any]) -> str:
    """Return a badge for a state's type (initial / final / normal)."""
    if state.get("initial"):
        return badge("initial", "initial")
    if state.get("final"):
        return badge("final", "final")
    return badge("state", "normal")


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a styled section header with optional subtitle."""
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-header"><h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Diagram container
# ---------------------------------------------------------------------------


def diagram_container(graph_string: str, label: str | None = None) -> None:
    """Render a Graphviz chart inside a styled card."""
    label_html = f'<div class="diagram-label">{label}</div>' if label else ""
    st.markdown(f'<div class="diagram-wrap">{label_html}</div>', unsafe_allow_html=True)
    st.graphviz_chart(graph_string, use_container_width=True)


# ---------------------------------------------------------------------------
# Styled HTML tables
# ---------------------------------------------------------------------------


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a raw HTML table string."""
    ths = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f"<td>{cell}</td>" for cell in row)
        body += f"<tr>{tds}</tr>\n"
    return (
        '<table class="styled-table">'
        f"<thead><tr>{ths}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_states_table(states: list[dict[str, Any]]) -> None:
    """Render a styled states table with badge pills."""
    if not states:
        st.markdown(
            '<div class="empty-state">No states defined</div>', unsafe_allow_html=True
        )
        return

    headers = ["Name", "ID", "Type", "Enter Hook", "Exit Hook"]
    rows: list[list[str]] = []
    for s in states:
        rows.append(
            [
                f"<strong>{s.get('name', s.get('id', ''))}</strong>",
                f'<span class="cell-mono">{s.get("id", "")}</span>',
                state_type_badge(s),
                f'<span class="cell-mono">{s.get("enter_hook") or "—"}</span>',
                f'<span class="cell-mono">{s.get("exit_hook") or "—"}</span>',
            ]
        )
    st.markdown(_html_table(headers, rows), unsafe_allow_html=True)


def render_transitions_table(transitions: list[dict[str, Any]]) -> None:
    """Render a styled transitions table."""
    if not transitions:
        st.markdown(
            '<div class="empty-state">No transitions defined</div>',
            unsafe_allow_html=True,
        )
        return

    headers = ["Event", "Source", "Target", "Type", "Guards"]
    rows: list[list[str]] = []
    for tr in transitions:
        ev = tr.get("event", "")
        is_internal = tr.get("internal", False)
        type_badge = (
            badge("internal", "internal")
            if is_internal
            else badge("transition", "primary")
        )

        guards: list[str] = []
        for g in tr.get("guards", []):
            guards.append(g)
        for u in tr.get("unless_guards", []):
            guards.append(f"!{u}")
        guard_str = (
            f'<span class="cell-mono">{", ".join(guards)}</span>' if guards else "—"
        )

        rows.append(
            [
                f'<span class="cell-mono"><strong>{ev}</strong></span>',
                f'<span class="cell-mono">{tr.get("source", "")}</span>',
                f'<span class="cell-mono">{tr.get("target", "")}</span>',
                type_badge,
                guard_str,
            ]
        )
    st.markdown(_html_table(headers, rows), unsafe_allow_html=True)


def render_events_table(events: list[dict[str, Any]], kind: str = "publishing") -> None:
    """Render a styled publishing or listening events table."""
    if not events:
        st.markdown(
            f'<div class="empty-state">No {kind} events</div>', unsafe_allow_html=True
        )
        return

    if kind == "publishing":
        headers = ["Name", "Tag", "Event Class"]
        rows = [
            [
                f"<strong>{e.get('name', '')}</strong>",
                badge(str(e.get("tag", "—")), "event") if e.get("tag") else "—",
                f'<span class="cell-mono">{e.get("event_class") or "—"}</span>',
            ]
            for e in events
        ]
    else:
        headers = ["Name", "Event Tag", "Publisher"]
        rows = [
            [
                f"<strong>{e.get('name', '')}</strong>",
                badge(str(e.get("event_tag", "—")), "event")
                if e.get("event_tag")
                else "—",
                f'<span class="cell-mono">{e.get("publisher_class") or "—"}</span>',
            ]
            for e in events
        ]
    st.markdown(_html_table(headers, rows), unsafe_allow_html=True)


def render_wiring_table(wiring: list[dict[str, Any]]) -> None:
    """Render the cross-agent event wiring table."""
    if not wiring:
        st.markdown(
            '<div class="empty-state">No event wiring detected</div>',
            unsafe_allow_html=True,
        )
        return

    headers = ["Publisher", "Listener", "Event Tag", "External"]
    rows = [
        [
            f"<strong>{w.get('publisher_agent') or '—'}</strong>",
            f"<strong>{w.get('listener_agent', '')}</strong>",
            badge(str(w.get("event_tag", "—")), "event") if w.get("event_tag") else "—",
            f'<span class="cell-mono">{w.get("external_publisher") or "—"}</span>',
        ]
        for w in wiring
    ]
    st.markdown(_html_table(headers, rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Schema property table
# ---------------------------------------------------------------------------


def render_schema_properties(
    json_schema: dict[str, Any], title: str = "Properties"
) -> None:
    """Parse a Pydantic JSON schema and render as a readable property table."""
    props = json_schema.get("properties", {})
    required_fields = set(json_schema.get("required", []))

    if not props:
        st.markdown(
            f'<div class="empty-state">No {title.lower()} defined</div>',
            unsafe_allow_html=True,
        )
        return

    headers = ["Field", "Type", "Required", "Default", "Description"]
    rows: list[list[str]] = []
    for name, spec in props.items():
        field_type = _resolve_type(spec)
        is_required = name in required_fields
        req_badge = (
            badge("required", "initial") if is_required else badge("optional", "normal")
        )
        default = spec.get("default", "—")
        if default is None:
            default = '<span class="cell-muted">None</span>'
        elif default == "—":
            default = "—"
        else:
            default = f'<span class="cell-mono">{default}</span>'
        desc = spec.get("description", "—")
        rows.append(
            [
                f'<span class="cell-mono"><strong>{name}</strong></span>',
                f'<span class="cell-mono">{field_type}</span>',
                req_badge,
                default,
                desc,
            ]
        )
    st.markdown(_html_table(headers, rows), unsafe_allow_html=True)


def _resolve_type(spec: dict[str, Any]) -> str:
    """Extract a human-readable type string from a JSON schema property."""
    if "anyOf" in spec:
        types = [_resolve_type(s) for s in spec["anyOf"]]
        return " | ".join(types)
    if "allOf" in spec:
        types = [_resolve_type(s) for s in spec["allOf"]]
        return " & ".join(types)
    if "$ref" in spec:
        ref: str = spec["$ref"].rsplit("/", 1)[-1]
        return ref
    type_str: str = spec.get("type", "any")
    if type_str == "array":
        items = spec.get("items", {})
        inner = _resolve_type(items)
        return f"list[{inner}]"
    if type_str == "object":
        return "dict"
    return str(type_str)
