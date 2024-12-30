from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from statemachine import State, StateMachine
from statemachine.factory import StateMachineMetaclass
from statemachine.transition_list import TransitionList

from .publishing import StateAgentPublishing

if TYPE_CHECKING:
    from .agent import Agent


T = TypeVar("T", bound="AgentStateMachine")


class AgentStateMachineMeta(StateMachineMetaclass):
    """Metaclass for AgentStateMachine that ensures proper state machine inheritance."""

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        # Don't modify the base class
        if name == "AgentStateMachine":
            return super().__new__(mcs, name, bases, namespace)

        # Create state definitions if not already defined
        if "starting" not in namespace:
            namespace["starting"] = State("Starting", initial=True)
        if "idle" not in namespace:
            namespace["idle"] = State("Idle")
        if "end" not in namespace:
            namespace["end"] = State("End", final=True)

        # Create transition definitions if not already defined
        if "start" not in namespace:
            namespace["start"] = namespace["starting"].to(namespace["idle"])
        if "stop" not in namespace:
            namespace["stop"] = namespace["starting"].to(namespace["end"]) | namespace[
                "idle"
            ].to(namespace["end"])

        return super().__new__(mcs, name, bases, namespace)


class AgentStateMachine(StateMachine, metaclass=AgentStateMachineMeta):
    """Base class for all agent state machines.

    This class provides the basic state machine structure that all agents should inherit from.
    Child classes can override states and transitions by defining their own, or inherit the defaults.
    """

    # === States ===

    starting: ClassVar[State]
    idle: ClassVar[State]
    end: ClassVar[State]

    # === Transitions ===

    start: ClassVar[TransitionList]
    stop: ClassVar[TransitionList]

    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self._agent = agent
        self._should_stop = False

    # === State actions ===

    def on_enter_end(self) -> None:
        self.agent.teardown()

    # === Transitions Actions ===

    def before_transition(
        self, event: str, state: str, source: str, target: str
    ) -> None:
        if source == target:
            return

        self.agent.logger().debug(
            f"[{self.agent.__class__.__name__}:StateMachine] ({event}) {source} -> {target}"
        )

    @abstractmethod
    def after_transition(self, event: str, state: str) -> None:
        """Handle behavior after a transition.

        This method should be implemented in child classes to handle post-transition behavior.
        Example implementation:
            self.agent.publishing.publish_transition(event, state)
        """
        raise NotImplementedError

    def on_start(self) -> None:
        self.agent.listening.subscribe()

    def on_stop(self) -> None:
        self._should_stop = True

    # === Conditions ===

    def should_stop(self) -> bool:
        return self._should_stop

    # === Properties ===

    @property
    def agent(self) -> "Agent":
        return self._agent

    # === Functions ===

    def safe_stop(self) -> None:
        """Safely stop the state machine if not already in final state."""
        if self.current_state.final:
            return
        self.stop(f=True)

    # === Diagrams ===


# Example of how to create a concrete state machine:
class BasicAgentStateMachine(AgentStateMachine):
    """A basic implementation of an agent state machine.

    This class inherits all the default states and transitions from AgentStateMachine.
    It only needs to implement the required after_transition method.
    """

    def after_transition(self, event: str, state: str) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
