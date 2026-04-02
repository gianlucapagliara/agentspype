"""Configuration page — create and validate agent YAML configs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from pydantic import ValidationError

from ..components import metric_card, render_metric_row, section_header
from ..theme import PRIMARY


def render_config_page(agent_classes: list[type]) -> None:
    """Render the configuration management page."""
    section_header("Configuration", "Create or validate agent YAML configs")

    agent_names = [cls.__name__ for cls in agent_classes]
    selected_name = st.selectbox("Agent", agent_names)
    if selected_name is None:
        return

    agent_class: Any = next(c for c in agent_classes if c.__name__ == selected_name)
    config_class = agent_class.definition.configuration_class

    # Field count metric
    schema = config_class.model_json_schema()
    field_count = len(schema.get("properties", {}))
    render_metric_row(
        [
            metric_card(
                "Config Fields", field_count, PRIMARY, sub=config_class.__name__
            ),
        ]
    )

    tab_new, tab_validate = st.tabs(["New Config", "Validate Config"])

    with tab_new:
        _render_new_config(agent_class, config_class, selected_name)

    with tab_validate:
        _render_validate_config(config_class)


def _render_new_config(agent_class: Any, config_class: Any, agent_name: str) -> None:
    """Render the new config form and YAML generator."""
    from pydantic_wizard.streamlit_ui.model_form import render_model_form
    from pydantic_wizard.validation import validate_and_fix

    from agentspype.runner.config.wizard import wrap_config_fields

    with st.container(border=True):
        with st.form("new_config_form"):
            data = render_model_form(config_class, key_prefix="new_")
            submitted = st.form_submit_button("Generate YAML")

    if submitted:
        instance = validate_and_fix(config_class, data)
        if instance is None:
            st.error("Validation failed. Check the field values above.")
        else:
            fqn = f"{agent_class.__module__}.{agent_class.__name__}"
            module_path, _, class_name = fqn.rpartition(".")
            full = wrap_config_fields(
                instance.model_dump(),
                agent_class=class_name,
                agent_path=module_path,
            )
            st.success("Valid configuration generated:")
            st.code(
                yaml.dump(full, default_flow_style=False, allow_unicode=True),
                language="yaml",
            )


def _render_validate_config(config_class: Any) -> None:
    """Render the config file validator."""
    from agentspype.runner.config.wizard import (
        extract_config_fields,
        load_agentspype_configs,
    )

    with st.container(border=True):
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
