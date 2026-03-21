"""Tests for RunnerConfig model."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentspype.runner.config.models import RunnerConfig


class TestRunnerConfig:
    def test_minimal(self) -> None:
        cfg = RunnerConfig(config_file=Path("agents.yaml"))
        assert cfg.config_file == Path("agents.yaml")
        assert cfg.log_level == "INFO"
        assert cfg.shutdown_timeout == 300.0
        assert cfg.instance_prefix is None

    def test_custom_values(self) -> None:
        cfg = RunnerConfig(
            config_file=Path("/etc/agents.yaml"),
            log_level="DEBUG",
            shutdown_timeout=60.0,
            instance_prefix="custom",
        )
        assert cfg.log_level == "DEBUG"
        assert cfg.shutdown_timeout == 60.0
        assert cfg.instance_prefix == "custom"

    def test_missing_config_file_raises(self) -> None:
        with pytest.raises(ValidationError):
            RunnerConfig()  # type: ignore[call-arg]
