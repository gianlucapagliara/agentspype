"""Tests for LocalConfigSource."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentspype.runner.config.local import LocalConfigSource
from agentspype.runner.config.source import ConfigSource, compute_config_hash


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    p = tmp_path / "agents.yaml"
    p.write_text("agent_name: TestAgent\nagent_path: mypackage/agent\n")
    return p


class TestLocalConfigSource:
    def test_implements_protocol(self, yaml_file: Path) -> None:
        source = LocalConfigSource(yaml_file)
        assert isinstance(source, ConfigSource)

    async def test_load_returns_config(self, yaml_file: Path) -> None:
        source = LocalConfigSource(yaml_file)
        result = await source.load()
        assert result.config == {
            "agent_name": "TestAgent",
            "agent_path": "mypackage/agent",
        }

    async def test_load_hash_prefixed(self, yaml_file: Path) -> None:
        source = LocalConfigSource(yaml_file)
        result = await source.load()
        expected = f"LOCAL_{compute_config_hash(result.config)}"
        assert result.config_hash == expected

    async def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        source = LocalConfigSource(p)
        result = await source.load()
        assert result.config == {}

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.yaml"
        source = LocalConfigSource(p)
        with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
            await source.load()
