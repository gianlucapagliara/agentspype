"""Tests for loader utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentspype.agent.agent import Agent
from agentspype.runner.config.loader import (
    generate_instance_prefix,
    load_agent_configs,
    resolve_agent_class,
)


class TestResolveAgentClass:
    def test_resolve_known_class(self) -> None:
        cls = resolve_agent_class("agentspype/agent/agent", "Agent")
        assert cls is Agent

    def test_resolve_with_dot_notation(self) -> None:
        cls = resolve_agent_class("agentspype.agent.agent", "Agent")
        assert cls is Agent

    def test_resolve_missing_module_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="Cannot import agent module"):
            resolve_agent_class("nonexistent.module.path", "Agent")

    def test_resolve_missing_class_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="has no class"):
            resolve_agent_class("agentspype.agent.agent", "NoSuchClass")


class TestGenerateInstancePrefix:
    def test_length(self) -> None:
        prefix = generate_instance_prefix(Path("agents.yaml"))
        assert len(prefix) == 10

    def test_hex_chars(self) -> None:
        prefix = generate_instance_prefix(Path("/some/path/config.yaml"))
        assert all(c in "0123456789abcdef" for c in prefix)


class TestLoadAgentConfigs:
    def test_single_config(self) -> None:
        config = {"agent_name": "Test", "agent_path": "pkg/agent"}
        result = load_agent_configs(config, "abc123")
        assert len(result) == 1
        assert result[0]["agent_name"] == "Test"

    def test_configuration_data_envelope(self) -> None:
        config = {
            "configuration_data": [
                {"agent_name": "A"},
                {"agent_name": "B"},
            ]
        }
        result = load_agent_configs(config, "prefix")
        assert len(result) == 2

    def test_instance_prefix_substitution(self) -> None:
        config = {"name": "$INSTANCE_PREFIX_myagent"}
        result = load_agent_configs(config, "abc")
        assert result[0]["name"] == "abc_myagent"

    def test_none_string_replaced(self) -> None:
        config = {"value": "None"}
        result = load_agent_configs(config, "x")
        assert result[0]["value"] is None
