import logging
from typing import TYPE_CHECKING, Any

from agentspype.agency import Agency
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition

if TYPE_CHECKING:
    from agentspype.agent.listening import AgentListening
    from agentspype.agent.publishing import AgentPublishing
    from agentspype.agent.state_machine import AgentStateMachine
    from agentspype.agent.status import AgentStatus


class Agent:
    _logger: logging.Logger | None = None

    # === Definition ===

    definition: AgentDefinition

    # === Initialization ===

    def __init__(self, configuration: AgentConfiguration | dict[str, Any]):
        self._configuration = (
            configuration
            if isinstance(configuration, AgentConfiguration)
            else self.definition.configuration_class(**configuration)
        )

        self._events_publishing = self.definition.events_publishing_class(self)
        self._events_listening = self.definition.events_listening_class(self)
        self._state_machine = self.definition.state_machine_class(self)
        self._status = self.definition.status_class()

        Agency.register_agent(self)

        self.initialize()

    def initialize(self) -> None:
        pass

    def teardown(self) -> None:
        self.listening.unsubscribe()
        Agency.deregister_agent(self)

    def clone(self) -> "Agent":
        return self.__class__(self.configuration)

    def __del__(self) -> None:
        try:
            self.teardown()
        except Exception:
            pass

    # === Class Methods ===

    @classmethod
    def logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger(cls.__name__)
        return cls._logger

    # === Properties ===

    @property
    def configuration(self) -> AgentConfiguration:
        return self._configuration

    @property
    def machine(self) -> "AgentStateMachine":
        return self._state_machine

    @property
    def listening(self) -> "AgentListening":
        return self._events_listening

    @property
    def publishing(self) -> "AgentPublishing":
        return self._events_publishing

    @property
    def status(self) -> "AgentStatus":
        return self._status
