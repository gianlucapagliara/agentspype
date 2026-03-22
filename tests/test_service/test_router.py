"""FastAPI TestClient tests for AgentRunnerRouter — requires processpype + httpx."""

from __future__ import annotations

import time
from typing import Any

import pytest

processpype = pytest.importorskip("processpype")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from processpype.core.models import ServiceState, ServiceStatus

from agentspype.service.events import QueueEventSubscriber
from agentspype.service.models import (
    AgentDetailModel,
    AgentSummaryModel,
    HealthModel,
    RunnerStatusModel,
)
from agentspype.service.router import AgentRunnerRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AGENT_SUMMARIES = [
    AgentSummaryModel(
        index=0,
        name="TestAgent",
        complete_name="PID:-|AID:1|TestAgent",
        is_initial=False,
        is_final=False,
        state="Idle",
    ),
]

_AGENT_DETAIL = AgentDetailModel(
    index=0,
    name="TestAgent",
    complete_name="PID:-|AID:1|TestAgent",
    is_initial=False,
    is_final=False,
    state="Idle",
    status={"key": "value"},
    configuration={"param": 42},
)

_RUNNER_STATUS = RunnerStatusModel(
    start_time=1000.0,
    uptime_seconds=10.0,
    agent_count=1,
    keep_running=False,
)

_HEALTH = HealthModel(status="ok", timestamp=time.time())


def _make_status() -> ServiceStatus:
    return ServiceStatus(
        state=ServiceState.RUNNING, error=None, metadata={}, is_configured=True
    )


async def _mock_stop_agent(index: int) -> dict[str, Any]:
    if index < 0 or index >= 1:
        raise IndexError(f"Agent {index} not found")
    return {"status": "stopped", "index": index}


def _mock_detail(index: int) -> AgentDetailModel:
    if index < 0 or index >= 1:
        raise IndexError(f"Agent {index} not found")
    return _AGENT_DETAIL


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient with a fully wired router."""
    bridge = QueueEventSubscriber()

    router = AgentRunnerRouter(
        name="test",
        get_status=_make_status,
        get_agents_summary=lambda: _AGENT_SUMMARIES,
        get_agent_detail=_mock_detail,
        stop_agent=_mock_stop_agent,
        get_runner_status=lambda: _RUNNER_STATUS,
        get_health=lambda: _HEALTH,
        event_subscriber=bridge,
    )

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_agents(client: TestClient) -> None:
    resp = client.get("/services/test/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "TestAgent"
    assert data[0]["state"] == "Idle"


def test_get_agent_detail(client: TestClient) -> None:
    resp = client.get("/services/test/agents/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["index"] == 0
    assert data["status"] == {"key": "value"}
    assert data["configuration"] == {"param": 42}


def test_get_agent_not_found(client: TestClient) -> None:
    resp = client.get("/services/test/agents/999")
    assert resp.status_code == 404


def test_stop_agent(client: TestClient) -> None:
    resp = client.post("/services/test/agents/0/stop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stopped"


def test_stop_agent_not_found(client: TestClient) -> None:
    resp = client.post("/services/test/agents/999/stop")
    assert resp.status_code == 404


def test_runner_status(client: TestClient) -> None:
    resp = client.get("/services/test/runner/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["start_time"] == 1000.0
    assert data["agent_count"] == 1


def test_health(client: TestClient) -> None:
    resp = client.get("/services/test/runner/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_service_status(client: TestClient) -> None:
    """The base ServiceRouter status endpoint still works."""
    resp = client.get("/services/test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "running"


def test_no_callbacks_returns_501() -> None:
    """When callbacks are None, endpoints return 501."""
    router = AgentRunnerRouter(
        name="empty",
        get_status=_make_status,
    )
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)

    assert c.get("/services/empty/agents").status_code == 501
    assert c.get("/services/empty/agents/0").status_code == 501
    assert c.post("/services/empty/agents/0/stop").status_code == 501
    assert c.get("/services/empty/runner/status").status_code == 501
    assert c.get("/services/empty/runner/health").status_code == 501
    assert c.get("/services/empty/events").status_code == 501
