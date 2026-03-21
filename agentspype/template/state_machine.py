from typing import TYPE_CHECKING

from statemachine import State

from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine

if TYPE_CHECKING:
    from agentspype.template.agent import TemplateAgent


class TemplateStateMachine(AgentStateMachine):
    if TYPE_CHECKING:
        agent: TemplateAgent

    # === States ===

    # Inherits: starting (initial), idle, end (final)
    # Add custom states here, e.g.:
    # running = State("Running")

    # === Transitions ===

    # Inherits: start (starting -> idle), stop (starting|idle -> end)
    # Add custom transitions here, e.g.:
    # run = idle.to(running)

    # === Transitions Actions ===

    def after_transition(self, event: str, state: State) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
