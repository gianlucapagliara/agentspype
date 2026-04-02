"""Custom CSS theme for the AgentsPype Streamlit UI.

Colors are derived from ``agentspype.visualization.theme.Theme`` to keep
the dashboard visually coherent with the generated Graphviz diagrams.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette (mirrors visualization/theme.py)
# ---------------------------------------------------------------------------
PRIMARY = "#6c8ebf"
PRIMARY_BG = "#dae8fc"
SUCCESS = "#82b366"
SUCCESS_BG = "#d5e8d4"
DANGER = "#b85450"
DANGER_BG = "#f8cecc"
WARNING = "#d6b656"
WARNING_BG = "#fff2cc"
NEUTRAL = "#566573"
NEUTRAL_BG = "#d5d8dc"
PURPLE = "#9673a6"
PURPLE_BG = "#e1d5e7"
TEAL = "#1abc9c"
TEAL_BG = "#d1f2eb"
TEXT_DARK = "#2d3436"
TEXT_MUTED = "#636e72"
BORDER_LIGHT = "#e0e4e8"
PAGE_BG = "#f8f9fb"

_CSS = f"""
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] {{
    background-color: {PAGE_BG};
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #ffffff 0%, {PRIMARY_BG}44 100%);
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    font-size: 0.92rem;
}}

/* ---- Metric cards ---- */
.metric-row {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}}
.metric-card {{
    flex: 1;
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    border-left: 4px solid {NEUTRAL};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.metric-card .metric-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    margin-bottom: 0.2rem;
}}
.metric-card .metric-value {{
    font-size: 1.65rem;
    font-weight: 700;
    color: {TEXT_DARK};
    line-height: 1.2;
}}
.metric-card .metric-sub {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    margin-top: 0.15rem;
}}

/* ---- Badges ---- */
.badge {{
    display: inline-block;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.5;
    white-space: nowrap;
}}
.badge-initial {{
    background: {SUCCESS_BG};
    color: {SUCCESS};
    border: 1px solid {SUCCESS};
}}
.badge-final {{
    background: {DANGER_BG};
    color: {DANGER};
    border: 1px solid {DANGER};
}}
.badge-normal {{
    background: {NEUTRAL_BG};
    color: {NEUTRAL};
    border: 1px solid {NEUTRAL};
}}
.badge-event {{
    background: {PURPLE_BG};
    color: {PURPLE};
    border: 1px solid {PURPLE};
}}
.badge-internal {{
    background: {WARNING_BG};
    color: {WARNING};
    border: 1px solid {WARNING};
}}
.badge-primary {{
    background: {PRIMARY_BG};
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
}}
.badge-teal {{
    background: {TEAL_BG};
    color: {TEAL};
    border: 1px solid {TEAL};
}}

/* ---- Section header ---- */
.section-header {{
    border-bottom: 2px solid {PRIMARY_BG};
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}}
.section-header h2 {{
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: {TEXT_DARK};
}}
.section-header .section-sub {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
}}

/* ---- Styled table ---- */
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}
.styled-table th {{
    text-align: left;
    padding: 0.55rem 0.75rem;
    background: {PRIMARY_BG};
    color: {TEXT_DARK};
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 2px solid {PRIMARY};
}}
.styled-table td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid {BORDER_LIGHT};
    color: {TEXT_DARK};
    vertical-align: middle;
}}
.styled-table tr:hover td {{
    background: {PAGE_BG};
}}
.styled-table .cell-muted {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
}}
.styled-table .cell-mono {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
}}

/* ---- Diagram container ---- */
.diagram-wrap {{
    background: #ffffff;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}}
.diagram-label {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {TEXT_MUTED};
    margin-bottom: 0.5rem;
}}

/* ---- Empty state ---- */
.empty-state {{
    text-align: center;
    padding: 2rem 1rem;
    color: {TEXT_MUTED};
    font-size: 0.9rem;
}}

/* ---- Sidebar nav ---- */
.sidebar-version {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    padding: 0 0 0.5rem 0;
}}
.sidebar-section {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    margin-top: 0.75rem;
    margin-bottom: 0.25rem;
}}
</style>
"""


def inject_custom_css() -> None:
    """Inject the full custom CSS into the Streamlit page (call once)."""
    st.markdown(_CSS, unsafe_allow_html=True)
