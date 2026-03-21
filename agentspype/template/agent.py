from typing import TYPE_CHECKING

from agentspype.agent.agent import Agent
from agentspype.agent.definition import AgentDefinition
from agentspype.template.configuration import TemplateConfiguration
from agentspype.template.listening import TemplateListening
from agentspype.template.publishing import TemplatePublishing
from agentspype.template.state_machine import TemplateStateMachine
from agentspype.template.status import TemplateStatus


class TemplateAgent(Agent):
    # === Agent Definition ===

    definition = AgentDefinition(
        state_machine_class=TemplateStateMachine,
        events_listening_class=TemplateListening,
        events_publishing_class=TemplatePublishing,
        configuration_class=TemplateConfiguration,
        status_class=TemplateStatus,
    )

    # === Casting ===

    if TYPE_CHECKING:
        machine: TemplateStateMachine
        listening: TemplateListening
        publishing: TemplatePublishing
        configuration: TemplateConfiguration
        status: TemplateStatus

    # === Initialization ===

    def initialize(self) -> None:
        super().initialize()

    def teardown(self) -> None:
        super().teardown()

    # === Properties ===

    ...

    # === Management ===

    ...

    # === Execution ===

    ...
