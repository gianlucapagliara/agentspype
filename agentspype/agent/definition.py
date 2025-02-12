from pydantic import BaseModel, ConfigDict

from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import AgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus


class AgentDefinition(BaseModel):
    """Definition of an agent's components and configuration."""

    model_config = ConfigDict(frozen=True)

    state_machine_class: type[AgentStateMachine]
    events_listening_class: type[AgentListening]
    events_publishing_class: type[AgentPublishing]
    configuration_class: type[AgentConfiguration]
    status_class: type[AgentStatus]
