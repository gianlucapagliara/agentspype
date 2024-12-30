from pydantic import BaseModel, ConfigDict

from .configuration import AgentConfiguration
from .listening import AgentListening
from .publishing import AgentPublishing
from .state_machine import AgentStateMachine
from .status import AgentStatus


class AgentDefinition(BaseModel):
    """Definition of an agent's components and configuration."""

    model_config = ConfigDict(frozen=True)

    state_machine_class: type[AgentStateMachine]
    events_listening_class: type[AgentListening]
    events_publishing_class: type[AgentPublishing]
    configuration_class: type[AgentConfiguration]
    status_class: type[AgentStatus]
