from __future__ import annotations

import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from agentspype.agent.publishing import StateAgentPublishing
from agentspype.fsm import State, StateMachine
from agentspype.fsm.machine import StateMachineMeta

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent


T = TypeVar("T", bound="AgentStateMachine")


class AgentStateMachineMeta(StateMachineMeta):
    """Metaclass for AgentStateMachine that ensures proper state machine inheritance.

    This metaclass solves the state-sharing problem in FSM inheritance.
    When subclassing a concrete state machine, the parent's State objects would normally be
    shared (and mutated) across subclasses. This metaclass creates fresh State copies for
    each new class and remaps all transitions to reference the fresh copies.

    It also auto-creates the default ``starting``, ``idle``, ``end`` states and
    ``start``, ``stop`` transitions when they are not explicitly defined.
    """

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        """Create a new AgentStateMachine subclass with isolated state objects."""
        if name == "AgentStateMachine":
            return super().__new__(mcs, name, bases, namespace)

        cls = super().__new__(mcs, name, bases, namespace)

        # Validate: every non-final state must have a 'stop' event defined.
        # This forces developers to explicitly decide stop semantics per state.
        states: list[State] = getattr(cls, "_states", [])
        transition_map: dict[tuple[str, str], Any] = getattr(cls, "_transition_map", {})
        missing = [
            state.id
            for state in states
            if not state.final and (state.id, "stop") not in transition_map
        ]
        if missing:
            raise ValueError(
                f"{name}: All non-final states must define a 'stop' transition. "
                f"Missing 'stop' for: {', '.join(sorted(missing))}. "
                f"Each state must explicitly handle the stop signal "
                f"(e.g., state.to(end), state.to(cleanup), or state.to.itself(internal=True))."
            )

        return cls


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
    # (start, stop are created by the metaclass as _EventDescriptor instances)

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = weakref.ref(agent)
        self._strong_agent: Agent | None = None
        self._should_stop = False

    # === State actions ===

    def on_enter_end(self) -> None:
        # Pin a strong reference so the agent survives until the state-machine
        # transition fully completes (after_transition, etc.).  Without this,
        # Python 3.14's more aggressive GC can collect the agent as soon as
        # teardown() removes it from Agency's lists.
        self._strong_agent = self.agent
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
    def after_transition(self, event: str, state: State) -> None:
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
    def agent(self) -> Agent:
        agent = self._agent()
        if agent is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: Agent reference expired (garbage collected or torn down)"
            )
        return agent

    # === Functions ===

    def safe_start(self) -> None:
        if not self.current_state.initial:
            return
        self.send("start", f=True)

    def safe_stop(self) -> None:
        """Safely stop the state machine if not already in final state."""
        if self.current_state.final:
            return
        self.send("stop", f=True)


# Example of how to create a concrete state machine:
class BasicAgentStateMachine(AgentStateMachine):
    """A basic implementation of an agent state machine.

    This class inherits all the default states and transitions from AgentStateMachine.
    It only needs to implement the required after_transition method.
    """

    def after_transition(self, event: str, state: State) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
