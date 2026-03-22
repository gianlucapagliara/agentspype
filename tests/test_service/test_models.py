"""Tests for agentspype.service.models — pure Pydantic models, no processpype needed."""

from __future__ import annotations

import time

from agentspype.service.models import (
    AgentDetailModel,
    AgentSummaryModel,
    HealthModel,
    RunnerStatusModel,
)


def test_agent_summary_model() -> None:
    """AgentSummaryModel can be instantiated and serialized."""
    model = AgentSummaryModel(
        index=0,
        name="TestAgent",
        complete_name="PID:-|AID:123|TestAgent",
        is_initial=True,
        is_final=False,
        state="Starting",
    )
    assert model.index == 0
    assert model.name == "TestAgent"
    assert model.is_initial is True
    assert model.is_final is False

    data = model.model_dump()
    assert isinstance(data, dict)
    assert data["state"] == "Starting"


def test_agent_detail_model() -> None:
    """AgentDetailModel extends AgentSummaryModel with status and configuration."""
    model = AgentDetailModel(
        index=1,
        name="DetailAgent",
        complete_name="PID:-|AID:456|DetailAgent",
        is_initial=False,
        is_final=True,
        state="End",
        status={"key": "value"},
        configuration={"param": 42},
    )
    assert model.status == {"key": "value"}
    assert model.configuration == {"param": 42}
    # Inherits from AgentSummaryModel
    assert model.is_final is True

    data = model.model_dump()
    assert "status" in data
    assert "configuration" in data
    assert "index" in data  # inherited


def test_runner_status_model() -> None:
    """RunnerStatusModel can be instantiated and serialized."""
    model = RunnerStatusModel(
        start_time=1000.0,
        uptime_seconds=42.5,
        agent_count=3,
        keep_running=True,
    )
    assert model.start_time == 1000.0
    assert model.agent_count == 3
    assert model.keep_running is True

    data = model.model_dump()
    assert isinstance(data, dict)
    assert data["uptime_seconds"] == 42.5


def test_health_model() -> None:
    """HealthModel can be instantiated and serialized."""
    ts = time.time()
    model = HealthModel(status="ok", timestamp=ts)
    assert model.status == "ok"
    assert model.timestamp == ts

    data = model.model_dump()
    assert data["status"] == "ok"


def test_agent_summary_model_json_roundtrip() -> None:
    """AgentSummaryModel can be serialized to JSON and deserialized."""
    model = AgentSummaryModel(
        index=0,
        name="RoundTrip",
        complete_name="PID:-|AID:0|RoundTrip",
        is_initial=False,
        is_final=False,
        state="Idle",
    )
    json_str = model.model_dump_json()
    restored = AgentSummaryModel.model_validate_json(json_str)
    assert restored == model
