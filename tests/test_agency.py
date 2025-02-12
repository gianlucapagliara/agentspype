from collections.abc import Generator
from typing import Any

import pytest
from statemachine import State

from agentspype.agency import Agency
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus


# Mock classes for testing
class MockStateMachine(AgentStateMachine):
    # Define states
    starting = State("Starting", initial=True)
    idle = State("Idle")
    mock = State("mock")
    end = State("End", final=True)

    # Define transitions
    start_to_end = starting.to(end)
    start_to_idle = starting.to(idle)
    idle_to_mock = idle.to(mock)
    mock_to_end = mock.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self._should_stop = False

    def after_transition(self, event: str, state: State) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)


class MockListening(AgentListening):
    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self.subscribed = False
        self.unsubscribed = False

    def subscribe(self) -> None:
        self.subscribed = True

    def unsubscribe(self) -> None:
        self.unsubscribed = True


class MockPublishing(StateAgentPublishing):
    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self.published_events: list[tuple[Any, Any]] = []

    def publish(self, event_publication: Any, event_data: Any) -> None:
        self.published_events.append((event_publication, event_data))

    sm_transition_event = StateAgentPublishing.sm_transition_event


class MockAgent(Agent):
    """A test agent implementation for testing Agency functionality."""

    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=MockPublishing,
        events_listening_class=MockListening,
        state_machine_class=MockStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def test_agent() -> Generator[Agent, None, None]:
    """Fixture to create a test agent."""
    agent = MockAgent({})
    yield agent
    try:
        agent.teardown()
    except ValueError:
        pass  # Agent might already be torn down


@pytest.fixture(autouse=True)
def clear_agency() -> Generator[None, None, None]:
    """Clear the Agency's state before and after each test."""
    Agency.initialized_agents.clear()
    yield
    Agency.initialized_agents.clear()


def test_register_agent(test_agent: Agent) -> None:
    """Test agent registration with Agency."""
    assert test_agent in Agency.initialized_agents


def test_deregister_agent(test_agent: Agent) -> None:
    """Test agent deregistration from Agency."""
    Agency.deregister_agent(test_agent)
    assert test_agent not in Agency.initialized_agents


def test_get_active_agents(test_agent: Agent) -> None:
    """Test getting active agents from Agency."""
    assert test_agent in Agency.get_active_agents()
    test_agent.machine.start_to_end()
    assert test_agent not in Agency.get_active_agents()


def test_multiple_agents() -> None:
    """Test Agency handling multiple agents."""
    agents = [MockAgent({}) for _ in range(3)]
    assert len(Agency.initialized_agents) == 3
    assert all(agent in Agency.initialized_agents for agent in agents)


def test_agency_registration_idempotency(test_agent: Agent) -> None:
    """Test that registering the same agent multiple times has no effect."""
    initial_count = len(Agency.initialized_agents)
    Agency.register_agent(test_agent)
    assert len(Agency.initialized_agents) == initial_count


def test_agency_deregistration_idempotency(test_agent: Agent) -> None:
    """Test that deregistering an agent multiple times has no effect."""
    Agency.deregister_agent(test_agent)
    initial_count = len(Agency.initialized_agents)
    Agency.deregister_agent(test_agent)
    assert len(Agency.initialized_agents) == initial_count


def test_agency_empty_state() -> None:
    """Test Agency behavior with no registered agents."""
    Agency.initialized_agents.clear()
    assert len(Agency.initialized_agents) == 0
    assert len(Agency.get_active_agents()) == 0


def test_agency_logger_messages(test_agent: Agent) -> None:
    """Test that Agency logs registration and deregistration events."""
    Agency.deregister_agent(test_agent)
    Agency.register_agent(test_agent)
