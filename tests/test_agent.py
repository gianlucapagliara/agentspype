from collections.abc import Generator
from typing import Any

import pytest
from statemachine import State

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
    stop = starting.to(end) | idle.to(end) | mock.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self._should_stop = False
        self.__raise_error = False

    def set_raise_error(self, value: bool) -> None:
        self.__raise_error = value

    def before_transition(
        self, event: str, state: str, source: str, target: str
    ) -> None:
        if self.__raise_error:
            raise Exception("Test error")
        super().before_transition(event, state, source, target)

    def after_transition(self, event: str, state: State) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)


class OtherMockStateMachine(AgentStateMachine):
    # Define workflow-specific states
    starting = State("Starting", initial=True)
    processing = State("Processing")
    completed = State("Completed", final=True)
    failed = State("Failed", final=True)

    # Define workflow transitions
    start_processing = starting.to(processing)
    complete = processing.to(completed)
    fail = processing.to(failed)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self._should_stop = False

    def after_transition(self, event: str, state: State) -> None:
        pass


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

    def trigger_event(self, event_publication: Any, event_data: Any) -> None:
        self.published_events.append((event_publication, event_data))

    class StateMachineEvent:
        def __init__(self, event: str, state: State) -> None:
            self.event = event
            self.new_state = state

    # Override the event publication to use our mock event class
    sm_transition_event = type(
        "EventPublication",
        (),
        {
            "event_tag": StateAgentPublishing.Events.StateMachineTransition,
            "event_class": StateMachineEvent,
        },
    )()


# Define agents with different state machines
class OtherMockAgent(Agent):
    """An agent implementation with workflow states."""

    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=MockPublishing,
        events_listening_class=MockListening,
        state_machine_class=OtherMockStateMachine,
        status_class=AgentStatus,
    )


class MockAgent(Agent):
    """A test agent implementation."""

    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=MockPublishing,
        events_listening_class=MockListening,
        state_machine_class=MockStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def test_agent() -> Generator[MockAgent, None, None]:
    """Fixture to create a test agent."""
    agent = MockAgent({})
    yield agent
    try:
        agent.teardown()
    except ValueError:
        pass  # Agent might already be torn down


def test_independent_state_machines() -> None:
    """Test that different state machine implementations remain independent."""
    # Create agents with different state machines
    mock_agent = MockAgent({})
    other_mock_agent = OtherMockAgent({})

    # Test that each agent has its own state machine with different states
    assert isinstance(mock_agent.machine, MockStateMachine)
    assert isinstance(other_mock_agent.machine, OtherMockStateMachine)

    # Verify initial states (both start in "Starting" due to inheritance)
    assert mock_agent.machine.current_state == mock_agent.machine.starting
    assert other_mock_agent.machine.current_state == other_mock_agent.machine.starting
    # They have the same name but are different state instances
    assert (
        mock_agent.machine.current_state.name
        == other_mock_agent.machine.current_state.name
    )
    assert (
        mock_agent.machine.current_state is not other_mock_agent.machine.current_state
    )

    # Verify state sets are different (except for inherited states)
    mock_states = {state.name for state in mock_agent.machine.states}
    other_states = {state.name for state in other_mock_agent.machine.states}
    assert mock_states != other_states

    # Verify unique states in each machine
    assert "mock" in mock_states and "mock" not in other_states  # Mock-specific state
    assert (
        "Processing" in other_states and "Processing" not in mock_states
    )  # Other-specific state
    assert (
        "Completed" in other_states and "Completed" not in mock_states
    )  # Other-specific state
    assert (
        "Failed" in other_states and "Failed" not in mock_states
    )  # Other-specific state

    # Test transitions in mock_agent
    mock_agent.machine.start_to_idle(f=False)
    assert mock_agent.machine.current_state == mock_agent.machine.idle
    mock_agent.machine.idle_to_mock(f=False)
    assert mock_agent.machine.current_state == mock_agent.machine.mock
    mock_agent.machine.mock_to_end(f=False)
    assert mock_agent.machine.current_state == mock_agent.machine.end
    assert mock_agent.machine.current_state.final

    # Test transitions in other_mock_agent
    other_mock_agent.machine.start_processing(f=False)
    assert other_mock_agent.machine.current_state == other_mock_agent.machine.processing
    other_mock_agent.machine.complete(f=False)
    assert other_mock_agent.machine.current_state == other_mock_agent.machine.completed
    assert other_mock_agent.machine.current_state.final

    # Verify that transitions don't interfere
    assert not hasattr(mock_agent.machine, "start_processing")  # Other's transition
    assert not hasattr(other_mock_agent.machine, "start_to_idle")  # Mock's transition

    # Verify that state machines have different transitions by checking their available methods
    mock_methods = {
        name for name in dir(mock_agent.machine) if not name.startswith("_")
    }
    other_methods = {
        name for name in dir(other_mock_agent.machine) if not name.startswith("_")
    }
    assert mock_methods != other_methods
    assert "start_to_idle" in mock_methods and "start_to_idle" not in other_methods
    assert (
        "start_processing" in other_methods and "start_processing" not in mock_methods
    )


def test_agent_initialization() -> None:
    """Test agent initialization with both dict and AgentConfiguration."""
    # Test initialization with dict
    agent1 = MockAgent({})
    assert isinstance(agent1.configuration, AgentConfiguration)
    agent1.teardown()

    # Test initialization with AgentConfiguration
    config = AgentConfiguration()
    agent2 = MockAgent(config)
    assert agent2.configuration is config
    agent2.teardown()


def test_agent_properties(test_agent: MockAgent) -> None:
    """Test that all agent properties are properly initialized."""
    assert isinstance(test_agent.configuration, AgentConfiguration)
    assert isinstance(test_agent.machine, MockStateMachine)
    assert isinstance(test_agent.listening, MockListening)
    assert isinstance(test_agent.publishing, MockPublishing)
    assert isinstance(test_agent.status, AgentStatus)


def test_agent_clone(test_agent: MockAgent) -> None:
    """Test agent cloning functionality."""
    cloned_agent = test_agent.clone()
    try:
        assert isinstance(cloned_agent, MockAgent)
        assert cloned_agent is not test_agent
        assert isinstance(cloned_agent.configuration, AgentConfiguration)
        # For empty configurations, we don't need to compare values
        assert isinstance(cloned_agent.configuration, type(test_agent.configuration))
    finally:
        cloned_agent.teardown()


def test_agent_logger() -> None:
    """Test agent logger initialization and access."""
    logger = MockAgent.logger()
    assert logger is not None
    assert logger.name == "agent"
    # Test that subsequent calls return the same logger
    assert MockAgent.logger() is logger


def test_state_machine_error_handling(test_agent: MockAgent) -> None:
    """Test state machine error handling in processing loop."""
    # Enable error raising for this test
    test_agent.machine.set_raise_error(True)

    # The processing loop should catch the exception and return
    with pytest.raises(Exception, match="Test error"):
        test_agent.machine.start_to_idle()


def test_state_machine_transitions(test_agent: MockAgent) -> None:
    """Test state machine transitions and state changes."""
    # Test initial state
    assert test_agent.machine.current_state == test_agent.machine.starting

    # Test transition to idle
    test_agent.machine.start_to_idle()
    assert test_agent.machine.current_state == test_agent.machine.idle

    # Test transition to mock
    test_agent.machine.idle_to_mock()
    assert test_agent.machine.current_state == test_agent.machine.mock

    # Test transition to end
    test_agent.machine.mock_to_end()
    assert test_agent.machine.current_state == test_agent.machine.end
    assert test_agent.machine.current_state.final


def test_listening_lifecycle(test_agent: MockAgent) -> None:
    """Test agent listening subscription lifecycle."""
    listening = test_agent.listening
    assert isinstance(listening, MockListening)

    # Test initial state
    assert not listening.subscribed
    assert not listening.unsubscribed

    # Test subscribe
    listening.subscribe()
    assert listening.subscribed
    assert not listening.unsubscribed

    # Test unsubscribe
    listening.unsubscribe()
    assert listening.subscribed  # Subscribe state remains
    assert listening.unsubscribed


def test_publishing_events(test_agent: MockAgent) -> None:
    """Test agent publishing functionality."""
    publishing = test_agent.publishing
    assert isinstance(publishing, MockPublishing)

    # Initial state
    assert len(publishing.published_events) == 0

    # Test state machine transition event
    test_agent.machine.start_to_idle()
    assert len(publishing.published_events) == 1

    event_pub, event_data = publishing.published_events[0]
    assert event_pub == publishing.sm_transition_event
    assert event_data.event == "start_to_idle"
    assert event_data.new_state == test_agent.machine.idle


def test_agent_initialize_hook(test_agent: MockAgent) -> None:
    """Test agent initialization hook."""

    class InitTestAgent(MockAgent):
        def initialize(self) -> None:
            self.initialized = True

    agent = InitTestAgent({})
    try:
        assert hasattr(agent, "initialized")
        assert agent.initialized
    finally:
        agent.teardown()


def test_agent_safe_stop(test_agent: MockAgent) -> None:
    """Test agent safe stop functionality."""
    # Test stop when not in final state
    assert not test_agent.machine.current_state.final
    test_agent.machine.safe_stop()
    assert test_agent.machine.current_state.final

    # Test stop when already in final state (should do nothing)
    test_agent.machine.safe_stop()
    assert test_agent.machine.current_state.final


def test_state_machine_should_stop(test_agent: MockAgent) -> None:
    """Test state machine should_stop flag."""
    assert not test_agent.machine.should_stop()
    test_agent.machine.on_stop()
    assert test_agent.machine.should_stop()


def test_state_machine_before_transition(test_agent: MockAgent) -> None:
    """Test state machine before_transition hook."""
    # Same state transition should not trigger debug log
    test_agent.machine.before_transition("test", "mock", "mock", "mock")

    # Different state transition should trigger debug log
    test_agent.machine.before_transition("test", "mock", "idle", "mock")
