"""Tests for the configuration wizard adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agentspype.runner.config.wizard import (
    extract_config_fields,
    load_agentspype_configs,
    resolve_agent_fqn,
    wrap_config_fields,
)
from agentspype.template.agent import TemplateAgent
from agentspype.template.configuration import TemplateConfiguration

# ---------------------------------------------------------------------------
# resolve_agent_fqn
# ---------------------------------------------------------------------------


class TestResolveAgentFqn:
    def test_resolves_base_agent(self) -> None:
        agent_cls, config_cls = resolve_agent_fqn(
            "agentspype.template.agent.TemplateAgent"
        )
        assert agent_cls is TemplateAgent
        assert config_cls is TemplateConfiguration

    def test_bare_class_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="dotted path"):
            resolve_agent_fqn("Agent")

    def test_nonexistent_module_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            resolve_agent_fqn("nonexistent.module.path.Agent")

    def test_nonexistent_class_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            resolve_agent_fqn("agentspype.agent.agent.NoSuchClass")


# ---------------------------------------------------------------------------
# extract_config_fields
# ---------------------------------------------------------------------------


class TestExtractConfigFields:
    def test_extracts_routing_keys_and_fields(self) -> None:
        data: dict[str, Any] = {
            "agent_module_path": "mypackage/agents/my_agent",
            "agent_class_name": "MyAgent",
            "name": "test",
            "timeout": 30,
        }
        module_path, class_name, fields = extract_config_fields(data)
        assert module_path == "mypackage/agents/my_agent"
        assert class_name == "MyAgent"
        assert fields == {"name": "test", "timeout": 30}

    def test_missing_routing_keys_return_empty_strings(self) -> None:
        data: dict[str, Any] = {"name": "test"}
        module_path, class_name, fields = extract_config_fields(data)
        assert module_path == ""
        assert class_name == ""

    def test_preserves_all_non_routing_fields(self) -> None:
        data: dict[str, Any] = {
            "agent_module_path": "mod",
            "agent_class_name": "Cls",
            "a": 1,
            "b": [2, 3],
            "c": {"nested": True},
        }
        _, _, fields = extract_config_fields(data)
        assert set(fields.keys()) == {"a", "b", "c"}
        assert fields["b"] == [2, 3]
        assert fields["c"] == {"nested": True}


# ---------------------------------------------------------------------------
# wrap_config_fields
# ---------------------------------------------------------------------------


class TestWrapConfigFields:
    def test_wraps_data_with_routing_keys(self) -> None:
        result = wrap_config_fields(
            {"name": "test"},
            agent_module_path="pkg/agent",
            agent_class_name="MyAgent",
        )
        assert result["agent_module_path"] == "pkg/agent"
        assert result["agent_class_name"] == "MyAgent"
        assert result["name"] == "test"

    def test_routing_keys_appear_first(self) -> None:
        result = wrap_config_fields(
            {"z_field": 1, "a_field": 2},
            agent_module_path="mod",
            agent_class_name="Cls",
        )
        keys = list(result.keys())
        assert keys[0] == "agent_module_path"
        assert keys[1] == "agent_class_name"


# ---------------------------------------------------------------------------
# load_agentspype_configs
# ---------------------------------------------------------------------------


class TestLoadAgentspypeConfigs:
    def test_loads_envelope_with_list(self, tmp_path: Path) -> None:
        cfg = {
            "configuration_data": [
                {"agent_class_name": "A"},
                {"agent_class_name": "B"},
            ]
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        result = load_agentspype_configs(p)
        assert len(result) == 2
        assert result[0]["agent_class_name"] == "A"
        assert result[1]["agent_class_name"] == "B"

    def test_loads_envelope_with_single_dict(self, tmp_path: Path) -> None:
        cfg = {"configuration_data": {"agent_class_name": "A"}}
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        result = load_agentspype_configs(p)
        assert len(result) == 1
        assert result[0]["agent_class_name"] == "A"

    def test_loads_bare_flat_dict(self, tmp_path: Path) -> None:
        cfg = {"agent_class_name": "A", "name": "test"}
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        result = load_agentspype_configs(p)
        assert len(result) == 1
        assert result[0]["agent_class_name"] == "A"
        assert result[0]["name"] == "test"

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = load_agentspype_configs(p)
        assert result == [{}]


# ---------------------------------------------------------------------------
# save_agentspype_configs
# ---------------------------------------------------------------------------


class TestSaveAgentspypeConfigs:
    @pytest.fixture(autouse=True)
    def _require_pydantic_wizard(self) -> None:
        pytest.importorskip("pydantic_wizard")

    def _import_save(self):  # noqa: ANN202
        from agentspype.runner.config.wizard import save_agentspype_configs

        return save_agentspype_configs

    def test_saves_single_config_as_flat_dict(self, tmp_path: Path) -> None:
        save = self._import_save()
        configs = [{"agent_module_path": "pkg/agent", "agent_class_name": "A", "x": 1}]
        p = tmp_path / "out.yaml"
        save(configs, p)
        raw = yaml.safe_load(p.read_text())
        # Single agent should be flat dict, no envelope
        assert "configuration_data" not in raw
        assert raw["agent_class_name"] == "A"

    def test_saves_multiple_configs_with_envelope(self, tmp_path: Path) -> None:
        save = self._import_save()
        configs = [
            {"agent_class_name": "A"},
            {"agent_class_name": "B"},
        ]
        p = tmp_path / "out.yaml"
        save(configs, p)
        raw = yaml.safe_load(p.read_text())
        assert "configuration_data" in raw
        assert len(raw["configuration_data"]) == 2

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        save = self._import_save()
        p = tmp_path / "deep" / "nested" / "dir" / "out.yaml"
        save([{"agent_class_name": "A"}], p)
        assert p.exists()

    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        save = self._import_save()
        original = [
            {
                "agent_module_path": "pkg/agent",
                "agent_class_name": "MyAgent",
                "timeout": 30,
                "name": "test",
            }
        ]
        p = tmp_path / "roundtrip.yaml"
        save(original, p)
        loaded = load_agentspype_configs(p)
        assert len(loaded) == 1
        assert loaded[0]["agent_class_name"] == "MyAgent"
        assert loaded[0]["timeout"] == 30
        assert loaded[0]["name"] == "test"


# ---------------------------------------------------------------------------
# wizard_show
# ---------------------------------------------------------------------------


class TestWizardShow:
    @pytest.fixture(autouse=True)
    def _require_pydantic_wizard(self) -> None:
        pytest.importorskip("pydantic_wizard")

    def test_show_runs_without_error(self) -> None:
        from agentspype.runner.config.wizard import wizard_show

        # Should not raise; TemplateAgent has TemplateConfiguration with no required fields.
        wizard_show("agentspype.template.agent.TemplateAgent")


# ---------------------------------------------------------------------------
# wizard_validate
# ---------------------------------------------------------------------------


class TestWizardValidate:
    @pytest.fixture(autouse=True)
    def _require_pydantic_wizard(self) -> None:
        pytest.importorskip("pydantic_wizard")

    def test_validate_valid_config_returns_true(self, tmp_path: Path) -> None:
        from agentspype.runner.config.wizard import wizard_validate

        cfg = {
            "agent_module_path": "agentspype/template/agent",
            "agent_class_name": "TemplateAgent",
        }
        p = tmp_path / "valid.yaml"
        p.write_text(yaml.dump(cfg))
        result = wizard_validate(p)
        assert result is True
