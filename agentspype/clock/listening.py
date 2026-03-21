"""Clock event listening for time-driven agents.

Subscribes to clock start/tick/stop events and delegates to the agent's
``handle_clock_*`` methods.
"""

from typing import TYPE_CHECKING

try:
    from chronopype.clocks.base import (
        BaseClock,
        ClockStartEvent,
        ClockStopEvent,
        ClockTickEvent,
    )
except ImportError as _err:
    raise ImportError(
        "chronopype is required for clock support. "
        "Install it with: pip install agentspype[clock]"
    ) from _err

from eventspype.sub.subscription import PublicationSubscription

from agentspype.agent.listening import AgentListening
from agentspype.runner.runtime import get_runtime

if TYPE_CHECKING:
    from agentspype.clock.agent import ClockAgent


class ClockListening(AgentListening):
    """Listens to clock events and delegates to the agent's handlers."""

    # === Properties ===

    @property
    def clock_publisher(self) -> BaseClock:
        """Retrieve the clock instance from the runtime registry."""
        runtime = get_runtime()
        return runtime.get("clock", BaseClock)

    # === Casting ===

    @property
    def agent(self) -> "ClockAgent":
        return super().agent  # type: ignore[return-value]

    # === Callbacks ===

    def on_clock_start(self, event: ClockStartEvent) -> None:
        self.agent.handle_clock_start(event.timestamp)

    def on_clock_tick(self, event: ClockTickEvent) -> None:
        self.agent.handle_clock_tick(event.timestamp)

    def on_clock_stop(self, event: ClockStopEvent) -> None:
        self.agent.handle_clock_stop(event.timestamp)

    # === Subscriptions ===

    clock_start_subscription = PublicationSubscription(
        BaseClock,
        BaseClock.start_publication,
        on_clock_start,
        callback_with_event_info=False,
    )
    clock_tick_subscription = PublicationSubscription(
        BaseClock,
        BaseClock.tick_publication,
        on_clock_tick,
        callback_with_event_info=False,
    )
    clock_stop_subscription = PublicationSubscription(
        BaseClock,
        BaseClock.stop_publication,
        on_clock_stop,
        callback_with_event_info=False,
    )

    def clock_complete_subscription(self) -> None:
        """Subscribe to all clock events."""
        self.add_subscription(self.clock_start_subscription, self.clock_publisher)
        self.add_subscription(self.clock_tick_subscription, self.clock_publisher)
        self.add_subscription(self.clock_stop_subscription, self.clock_publisher)

    def clock_complete_unsubscription(self) -> None:
        """Unsubscribe from all clock events."""
        self.remove_subscription(self.clock_start_subscription, self.clock_publisher)
        self.remove_subscription(self.clock_tick_subscription, self.clock_publisher)
        self.remove_subscription(self.clock_stop_subscription, self.clock_publisher)

    def subscribe(self) -> None:
        self.clock_complete_subscription()

    def unsubscribe(self) -> None:
        self.clock_complete_unsubscription()
