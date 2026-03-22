"""Tests for loader utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentspype.agent.agent import Agent
from agentspype.runner.config.loader import (
    extract_config_fields,
    generate_instance_prefix,
    load_agent_configs,
    resolve_agent_class,
    resolve_agent_from_config,
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

    def test_list_at_root(self) -> None:
        config = [{"agent_class": "A"}, {"agent_class": "B"}]
        result = load_agent_configs(config, "pfx")
        assert len(result) == 2
        assert result[0]["agent_class"] == "A"
        assert result[1]["agent_class"] == "B"

    def test_list_at_root_with_substitution(self) -> None:
        config = [{"name": "$INSTANCE_PREFIX_x"}]
        result = load_agent_configs(config, "pfx")
        assert result[0]["name"] == "pfx_x"


class TestExtractConfigFields:
    def test_strips_new_routing_keys(self) -> None:
        raw = {
            "agent_class": "MyAgent",
            "agent_path": "pkg/agents/my_agent",
            "param_a": 1,
            "param_b": "hello",
        }
        result = extract_config_fields(raw)
        assert "agent_class" not in result
        assert "agent_path" not in result
        assert result == {"param_a": 1, "param_b": "hello"}

    def test_strips_legacy_routing_keys(self) -> None:
        raw = {
            "agent_module_path": "pkg.agents",
            "agent_class_name": "Foo",
            "agent_name": "foo",
            "agent_path": "pkg/agents",
            "agent_configuration": {"nested_key": 42},
            "top_level": True,
        }
        result = extract_config_fields(raw)
        assert "agent_module_path" not in result
        assert "agent_class_name" not in result
        assert "agent_name" not in result
        assert "agent_path" not in result
        assert "agent_configuration" not in result
        assert result["nested_key"] == 42
        assert result["top_level"] is True

    def test_agent_configuration_merges(self) -> None:
        raw = {
            "agent_class": "X",
            "agent_configuration": {"a": 1, "b": 2},
            "c": 3,
        }
        result = extract_config_fields(raw)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_empty_routing(self) -> None:
        raw = {"foo": "bar"}
        result = extract_config_fields(raw)
        assert result == {"foo": "bar"}


class TestResolveAgentFromConfig:
    def test_fallback_to_path_resolution(self) -> None:
        raw = {
            "agent_class": "MockAgent",
            "agent_path": "tests/test_agency",
        }
        agent_cls, config_cls = resolve_agent_from_config(raw)
        assert agent_cls.__name__ == "MockAgent"

    def test_legacy_format(self) -> None:
        raw = {
            "agent_class_name": "MockAgent",
            "agent_module_path": "tests.test_agency",
        }
        agent_cls, _ = resolve_agent_from_config(raw)
        assert agent_cls.__name__ == "MockAgent"

    def test_bl_agents_format(self) -> None:
        raw = {
            "agent_name": "MockAgent",
            "agent_path": "tests/test_agency",
        }
        agent_cls, _ = resolve_agent_from_config(raw)
        assert agent_cls.__name__ == "MockAgent"

    def test_missing_class_info_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain"):
            resolve_agent_from_config({"agent_path": "some/path"})

    def test_unresolvable_class_raises(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            resolve_agent_from_config({"agent_class": "NoSuchAgent"})
