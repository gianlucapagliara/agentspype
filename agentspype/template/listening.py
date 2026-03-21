from typing import TYPE_CHECKING

from agentspype.agent.listening import AgentListening

if TYPE_CHECKING:
    from agentspype.template.agent import TemplateAgent


class TemplateListening(AgentListening):
    if TYPE_CHECKING:
        agent: TemplateAgent

    # === Subscriptions ===

    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass
