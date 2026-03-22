"""Integration tests for AgentRunnerService — requires processpype."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

processpype = pytest.importorskip("processpype")

from statemachine import State

from agentspype.agency import Agency
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.runner.runner import AgentRunner
from agentspype.runner.runtime import set_runtime
from agentspype.service.configuration import AgentRunnerServiceConfiguration
from agentspype.service.service import AgentRunnerService

# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------


class _ServiceTestStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    stop = starting.to(end) | idle.to(end)

    def after_transition(self, event: str, state: State) -> None:
        pass


class _ServiceTestListening(AgentListening):
    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class _ServiceTestPublishing(StateAgentPublishing):
    def publish(self, event_publication: Any, event_data: Any) -> None:
        pass


class _ServiceTestConfig(AgentConfiguration):
    """Unique config class to avoid bidict collision with other test suites."""

    label: str = "service-test"


class _ServiceTestAgent(Agent):
    definition = AgentDefinition(
        configuration_class=_ServiceTestConfig,
        events_publishing_class=_ServiceTestPublishing,
        events_listening_class=_ServiceTestListening,
        state_machine_class=_ServiceTestStateMachine,
        status_class=AgentStatus,
    )


# ---------------------------------------------------------------------------
# Concrete service for testing
# ---------------------------------------------------------------------------


class _ConcreteService(AgentRunnerService):
    """Minimal concrete service that creates a simple AgentRunner."""

    def create_runner(self) -> AgentRunner:
        configs = [_ServiceTestConfig()]
        return AgentRunner(configs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_agency() -> Generator[None]:
    Agency.register_agent_class(_ServiceTestAgent)
    yield
    Agency.initialized_agents.clear()
    Agency._deactivating_agents.clear()
    set_runtime(None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_service_instantiation() -> None:
    """Service can be instantiated without starting."""
    service = _ConcreteService(name="test")
    assert service.name == "test"
    assert service._runner is None


def test_service_requires_configuration() -> None:
    """Service requires configuration before starting."""
    service = _ConcreteService(name="test")
    assert service.requires_configuration() is True


def test_service_create_runner_not_implemented() -> None:
    """Base class raises NotImplementedError."""
    service = AgentRunnerService.__new__(AgentRunnerService)
    with pytest.raises(NotImplementedError):
        service.create_runner()


@pytest.mark.asyncio
async def test_service_start_stop() -> None:
    """Service can start and stop with a runner."""
    service = _ConcreteService(name="test")
    config = AgentRunnerServiceConfiguration()
    service.configure(config)

    await service.start()

    assert service._runner is not None
    agents = service._runner.agents
    assert len(agents) == 1
    assert agents[0].is_initial is False  # should have transitioned past initial

    # Check data methods
    summaries = service.get_agents_summary()
    assert len(summaries) == 1
    assert summaries[0].name == "_ServiceTestAgent"

    detail = service.get_agent_detail(0)
    assert detail.index == 0

    health = service.get_health()
    assert health.status == "ok"

    runner_status = service.get_runner_status()
    assert runner_status.agent_count == 1

    await service.stop()
    assert service._runner is None

    # After stop, health should report error
    health = service.get_health()
    assert health.status == "error"


@pytest.mark.asyncio
async def test_service_stop_agent() -> None:
    """Service can stop a specific agent by index."""
    service = _ConcreteService(name="test")
    config = AgentRunnerServiceConfiguration()
    service.configure(config)
    await service.start()

    result = await service.stop_agent(0)
    assert result["status"] == "stopped"

    await service.stop()


@pytest.mark.asyncio
async def test_service_agent_index_out_of_range() -> None:
    """Out-of-range agent index raises IndexError."""
    service = _ConcreteService(name="test")
    config = AgentRunnerServiceConfiguration()
    service.configure(config)
    await service.start()

    with pytest.raises(IndexError):
        service.get_agent_detail(999)

    with pytest.raises(IndexError):
        await service.stop_agent(-1)

    await service.stop()


@pytest.mark.asyncio
async def test_service_runner_status_when_not_running() -> None:
    """Runner status returns defaults when runner is None."""
    service = _ConcreteService(name="test")
    status = service.get_runner_status()
    assert status.start_time == 0.0
    assert status.agent_count == 0
    assert status.keep_running is False
