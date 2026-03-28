"""Clock-aware state machine for time-driven agents.

Provides readiness-checking logic and the ``launch_tick`` driver method
that advances the state machine on each clock tick.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from agentspype.agent.state_machine import AgentStateMachine, BasicAgentStateMachine
from agentspype.fsm import State, TransitionList

if TYPE_CHECKING:
    from agentspype.clock.agent import ClockAgent


class ClockStateMachine(AgentStateMachine):
    """Abstract mixin for clock-driven state machines.

    Provides readiness-checking logic and the ``launch_tick`` driver.
    Concrete classes should inherit from both this and a complete agent
    state machine (see ``BasicClockStateMachine``).
    """

    # === States ===
    running: ClassVar[State]

    # === State actions ===

    def on_enter_idle(self) -> None:
        if self.is_ready:
            self.send("activate")
        else:
            self._on_enter_idle()

    @abstractmethod
    def _on_enter_idle(self) -> None:
        raise NotImplementedError

    def on_enter_running(self) -> None:
        if not self.is_ready:
            self.send("deactivate")
        else:
            self._on_enter_running()

    @abstractmethod
    def _on_enter_running(self) -> None:
        raise NotImplementedError

    # === Casting ===

    @property
    def agent(self) -> ClockAgent:
        return super().agent  # type: ignore[return-value]

    # === Properties ===

    @property
    def agent_ready_conditions(self) -> dict[str, bool]:
        """Return a dict of condition name to boolean.

        Override in subclasses to add domain-specific readiness conditions.
        The base implementation checks that at least one timestamp has been
        received.
        """
        return {"time": self.agent.last_timestamp > 0}

    @property
    def is_ready(self) -> bool:
        conditions = self.agent_ready_conditions
        self.agent.logger().debug(
            "[%s] Checking readiness: %s", self.agent.complete_name, conditions
        )
        return all(conditions.values())

    # === Core ===

    def launch_tick(self, timestamp: float) -> None:
        """Called by the agent on each clock tick.

        Updates the agent's last_timestamp (monotonically increasing) and,
        if the current state supports a ``tick`` transition, fires it.
        """
        self.agent.last_timestamp = timestamp

        if self.agent.last_timestamp != timestamp:
            return  # Skip if the timestamp was not updated (not monotonic)

        # Fire the tick event if the current state supports it
        if "tick" in self.current_state.transitions.unique_events:
            self.send("tick")

    def activate_initial_state(self) -> None:
        """Transition from idle to running when ready (called on clock start)."""
        if self.is_ready and self.current_state == self.idle:
            self.send("activate")


class BasicClockStateMachine(ClockStateMachine, BasicAgentStateMachine):
    """Concrete clock state machine with default states and transitions.

    Inherits ``starting``, ``idle``, ``end``, ``start``, ``stop`` from the
    metaclass and adds ``running``, ``activate``, ``deactivate``, and ``tick``.
    """

    def __init__(self, agent: ClockAgent) -> None:
        super().__init__(agent)

    # === States ===
    running: ClassVar[State] = State("Running")

    # === Transitions ===
    activate: ClassVar[TransitionList] = ClockStateMachine.idle.to(running)
    deactivate: ClassVar[TransitionList] = running.to(ClockStateMachine.idle)
    tick: ClassVar[TransitionList] = running.to.itself()
    stop: ClassVar[TransitionList] = running.to(ClockStateMachine.end)

    # === State action stubs ===

    def _on_enter_idle(self) -> None:
        pass

    def _on_enter_running(self) -> None:
        pass
