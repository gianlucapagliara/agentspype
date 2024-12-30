from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from eventspype.publishers.multi import MultiPublisher
from eventspype.publishers.publications import EventPublication
from statemachine import State

if TYPE_CHECKING:
    from .agent import Agent


class AgentPublishing(MultiPublisher):
    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self._agent = agent

    # === Properties ===

    @property
    def agent(self) -> "Agent":
        return self._agent


class StateAgentPublishing(AgentPublishing):
    # === Definitions ===

    class Events(Enum):
        StateMachineTransition = "sm_transition_event"

    @dataclass
    class StateMachineEvent:
        event: Any
        new_state: State

    # === Publications ===

    sm_transition_event = EventPublication(
        event_tag=Events.StateMachineTransition, event_class=StateMachineEvent
    )

    # === Events ===

    def publish_transition(self, event: Any, state: State) -> None:
        """
        Publishes a state machine transition event.

        It is recommended to call this method from the state machine.
        Example:
            def after_transition(self, event, state):
                self.agent.publishing.publish_transition(event, state)
        """
        self.trigger_event(
            self.sm_transition_event,
            self.sm_transition_event.event_class(event, state),
        )
