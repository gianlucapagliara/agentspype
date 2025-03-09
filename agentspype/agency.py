import asyncio
import logging
from typing import TYPE_CHECKING

from bidict import bidict

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent
    from agentspype.agent.configuration import AgentConfiguration


class Agency:
    initialized_agents: list["Agent"] = []
    _deactivating_agents: list["Agent"] = []
    _agent_to_configuration: bidict[type["Agent"], type["AgentConfiguration"]] = (
        bidict()
    )
    _logger: logging.Logger = logging.getLogger("agentspype.agency")

    @classmethod
    def register_agent(cls, agent: "Agent") -> None:
        """Register an initialized agent."""
        if agent not in cls.initialized_agents:
            cls._logger.info(f"[Agency] Registered: {agent.__class__.__name__}")
            cls.initialized_agents.append(agent)

    @classmethod
    def deregister_agent(cls, agent: "Agent") -> None:
        """Deregister an initialized agent."""
        if agent in cls.initialized_agents:
            cls._logger.info(f"[Agency] Deregistered: {agent.__class__.__name__}")
            cls.initialized_agents.remove(agent)
            cls._deactivating_agents.append(agent)

            async def cleanup() -> None:
                await asyncio.sleep(0)
                cls._deactivating_agents.remove(agent)

            asyncio.ensure_future(cleanup())

    @classmethod
    def get_active_agents(cls) -> list["Agent"]:
        """Get a list of all active agents."""
        return list(cls.initialized_agents)

    @classmethod
    def register_agent_class(cls, agent_class: type["Agent"]) -> None:
        """Register an agent class from its configuration."""
        if agent_class in cls._agent_to_configuration:
            return

        cls._agent_to_configuration[agent_class] = (
            agent_class.definition.configuration_class
        )

    @classmethod
    def deregister_agent_class(cls, agent_class: type["Agent"]) -> None:
        """Deregister an agent class."""
        if agent_class not in cls._agent_to_configuration:
            return

        del cls._agent_to_configuration[agent_class]

    @classmethod
    def get_agent_from_configuration(
        cls, configuration: "AgentConfiguration"
    ) -> "Agent":
        """Get an agent from its configuration."""
        if type(configuration) not in cls._agent_to_configuration.inverse:
            raise ValueError(
                f"No agent class registered for configuration {type(configuration)}"
            )

        agent_class = cls._agent_to_configuration.inverse[type(configuration)]
        return agent_class(configuration)
