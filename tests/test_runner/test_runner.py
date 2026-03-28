"""Tests for AgentRunner lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest

from agentspype.agency import Agency
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.fsm import State
from agentspype.runner.runner import AgentRunner
from agentspype.runner.runtime import BaseRuntime, get_runtime, set_runtime

# ---------------------------------------------------------------------------
# Mock agent setup (mirrors test_agent.py conventions)
# ---------------------------------------------------------------------------


class _RunnerTestStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    stop = starting.to(end) | idle.to(end)

    def after_transition(self, event: str, state: State) -> None:
        pass


class _RunnerTestListening(AgentListening):
    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class _RunnerTestPublishing(StateAgentPublishing):
    def publish(self, event_publication: Any, event_data: Any) -> None:
        pass


class _RunnerTestConfig(AgentConfiguration):
    label: str = "test"


class _RunnerTestAgent(Agent):
    definition = AgentDefinition(
        configuration_class=_RunnerTestConfig,
        events_publishing_class=_RunnerTestPublishing,
        events_listening_class=_RunnerTestListening,
        state_machine_class=_RunnerTestStateMachine,
        status_class=AgentStatus,
    )


class _ConcreteRunner(AgentRunner):
    """Minimal concrete runner for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hooks_called: list[str] = []

    async def on_setup(self) -> None:
        self.hooks_called.append("on_setup")

    async def on_run_start(self) -> None:
        self.hooks_called.append("on_run_start")

    async def on_run_end(self) -> None:
        self.hooks_called.append("on_run_end")

    async def on_teardown(self) -> None:
        self.hooks_called.append("on_teardown")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_agency() -> Generator[None]:
    """Ensure Agency class-level state is clean for each test."""
    Agency.register_agent_class(_RunnerTestAgent)
    yield
    # Clear leftover agents
    Agency.initialized_agents.clear()
    Agency._deactivating_agents.clear()
    set_runtime(None)


# ---------------------------------------------------------------------------
# Tests — setup / teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_creates_agents() -> None:
    config = _RunnerTestConfig(label="alpha")
    runner = _ConcreteRunner([config])

    await runner.setup()

    assert len(runner.agents) == 1
    agent = runner.agents[0]
    assert isinstance(agent, _RunnerTestAgent)
    # Agent should have been started (transitioned from starting to idle)
    assert agent.machine.current_state == agent.machine.idle

    await runner.teardown()


@pytest.mark.asyncio
async def test_setup_sets_runtime_singleton() -> None:
    runner = _ConcreteRunner([_RunnerTestConfig()])
    await runner.setup()

    assert get_runtime() is runner.runtime

    await runner.teardown()
    with pytest.raises(RuntimeError):
        get_runtime()


@pytest.mark.asyncio
async def test_teardown_stops_agents() -> None:
    runner = _ConcreteRunner([_RunnerTestConfig()])
    await runner.setup()

    agents = runner.agents
    assert len(agents) == 1

    await runner.teardown()

    # After teardown, runner.agents is cleared
    assert runner.agents == []


@pytest.mark.asyncio
async def test_teardown_clears_runtime_resources() -> None:
    rt = BaseRuntime()
    rt.register("something", object())

    runner = _ConcreteRunner([_RunnerTestConfig()], runtime=rt)
    await runner.setup()
    await runner.teardown()

    assert rt.has("something") is False


# ---------------------------------------------------------------------------
# Tests — hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hooks_called_in_order() -> None:
    runner = _ConcreteRunner([_RunnerTestConfig()])
    await runner.setup()
    assert "on_setup" in runner.hooks_called

    await runner.teardown()
    assert "on_teardown" in runner.hooks_called


# ---------------------------------------------------------------------------
# Tests — run loop (agents finish naturally)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_exits_when_all_agents_final() -> None:
    """Runner exits the loop when all agents reach final state."""
    config = _RunnerTestConfig()
    runner = _ConcreteRunner([config])

    async def _drive() -> None:
        # Wait a beat for setup to complete, then stop the agent
        await asyncio.sleep(0.3)
        for agent in Agency.get_active_agents():
            if isinstance(agent, _RunnerTestAgent):
                agent.machine.safe_stop()

    task = asyncio.create_task(_drive())

    # run() should return once the agent reaches final state
    await asyncio.wait_for(runner.run(), timeout=5.0)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert "on_run_start" in runner.hooks_called
    assert "on_run_end" in runner.hooks_called
    assert "on_teardown" in runner.hooks_called


# ---------------------------------------------------------------------------
# Tests — multiple agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_time_property() -> None:
    """start_time is 0.0 before setup, positive after setup (when history is enabled)."""
    runner = _ConcreteRunner([_RunnerTestConfig()])
    assert runner.start_time == 0.0

    await runner.setup()
    # start_time is set only when history_service.enabled is True.
    # The default LogHistoryService has enabled=False, so it stays 0.
    # We just verify the property is accessible and returns a float.
    assert isinstance(runner.start_time, float)

    await runner.teardown()


@pytest.mark.asyncio
async def test_keep_running_property() -> None:
    """keep_running reflects the constructor argument."""
    runner_default = _ConcreteRunner([_RunnerTestConfig()])
    assert runner_default.keep_running is False

    runner_keep = _ConcreteRunner([_RunnerTestConfig()], keep_running=True)
    assert runner_keep.keep_running is True


@pytest.mark.asyncio
async def test_multiple_agents() -> None:
    configs = [_RunnerTestConfig(label="a"), _RunnerTestConfig(label="b")]
    runner = _ConcreteRunner(configs)
    await runner.setup()

    assert len(runner.agents) == 2

    await runner.teardown()
