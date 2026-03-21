"""Clock-aware agent base class for time-driven agents.

Provides the core clock event handling hooks and monotonically increasing
timestamp tracking.
"""

from typing import TYPE_CHECKING, Any

from agentspype.agent.agent import Agent
from agentspype.clock.configuration import ClockAgentConfiguration

if TYPE_CHECKING:
    from agentspype.clock.listening import ClockListening
    from agentspype.clock.publishing import ClockAgentPublishing
    from agentspype.clock.state_machine import ClockStateMachine


class ClockAgent(Agent):
    """Base class for agents driven by a clock.

    Subclass this to create agents that react to clock start, tick, and stop
    events.  The ``last_timestamp`` property is updated monotonically on each
    tick.
    """

    # === Initialization ===

    def initialize(self) -> None:
        super().initialize()
        self._last_timestamp: float = 0

    # === Casting ===

    @property
    def machine(self) -> "ClockStateMachine":
        return super().machine  # type: ignore[return-value]

    @property
    def listening(self) -> "ClockListening":
        return super().listening

    @property
    def publishing(self) -> "ClockAgentPublishing":
        return super().publishing

    # === Properties ===

    @property
    def last_timestamp(self) -> float:
        """The most recent timestamp received from the clock."""
        return self._last_timestamp

    @last_timestamp.setter
    def last_timestamp(self, value: float) -> None:
        """Update the timestamp (monotonically increasing only)."""
        if value > self._last_timestamp:
            self._last_timestamp = value

    @property
    def configuration(self) -> ClockAgentConfiguration:
        return super().configuration  # type: ignore[return-value]

    # === Clock event hooks ===

    def handle_clock_start(self, timestamp: float) -> None:
        """Called when the clock starts.

        Override in subclasses to perform setup on clock start.
        Default behaviour: activate the initial state in the state machine.
        """
        self.machine.activate_initial_state()

    def handle_clock_tick(self, timestamp: float) -> None:
        """Called on each clock tick.

        Override in subclasses to add custom tick logic.
        Default behaviour: forward to the state machine's ``launch_tick``.
        """
        self.machine.launch_tick(timestamp)

    def handle_clock_stop(self, timestamp: float) -> None:
        """Called when the clock stops.

        Override in subclasses to perform cleanup on clock stop.
        Default behaviour: safely stop the state machine.
        """
        self.machine.safe_stop()

    # === Components ===

    def get_components(self) -> list[Any]:
        """Return sub-components for visualization.

        Override in subclasses to expose domain-specific components.
        """
        return []
