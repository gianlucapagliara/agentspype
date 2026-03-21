from agentspype.agent.publishing import StateAgentPublishing


class ClockAgentPublishing(StateAgentPublishing):
    """Publishing for clock-driven agents.

    Reuses StateAgentPublishing for state machine transition events.
    Subclass to add domain-specific publications.
    """
