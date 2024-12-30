from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent.agent import Agent


class Agency:
    initialized_agents: list["Agent"] = []

    @classmethod
    def register_agent(cls, agent: "Agent") -> None:
        """Register an initialized agent."""
        agent.logger().info(f"[Agency] Registered: {agent.__class__.__name__}")
        cls.initialized_agents.append(agent)

    @classmethod
    def deregister_agent(cls, agent: "Agent") -> None:
        """Deregister an initialized agent."""
        agent.logger().info(f"[Agency] Deregistered: {agent.__class__.__name__}")
        cls.initialized_agents.remove(agent)

    @classmethod
    def get_active_agents(cls) -> list["Agent"]:
        """Get a list of all active agents."""
        return [
            agent
            for agent in cls.initialized_agents
            if not agent.machine.current_state.final
        ]
