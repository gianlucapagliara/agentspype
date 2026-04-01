"""Tests for agentspype.ui.discovery."""

from __future__ import annotations

from typing import Any

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import BasicAgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.ui.discovery import discover_agent_classes

_DiscoverySM = BasicAgentStateMachine


class _DiscoveryPublishing(StateAgentPublishing):
    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def publish(self, event_publication: Any, event_data: Any) -> None:
        pass

    sm_transition_event = StateAgentPublishing.sm_transition_event


class _DiscoveryListening(AgentListening):
    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class _DiscoveryAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=_DiscoveryPublishing,
        events_listening_class=_DiscoveryListening,
        state_machine_class=_DiscoverySM,
        status_class=AgentStatus,
    )


class TestDiscovery:
    def test_discovers_agent_in_this_module(self) -> None:
        classes = discover_agent_classes(["tests.test_ui.test_discovery"])
        names = {cls.__name__ for cls in classes}
        assert "_DiscoveryAgent" in names

    def test_returns_empty_for_nonexistent_module(self) -> None:
        classes = discover_agent_classes(["nonexistent.module.xyz"])
        assert classes == []

    def test_deduplicates_across_repeated_modules(self) -> None:
        classes = discover_agent_classes(
            ["tests.test_ui.test_discovery", "tests.test_ui.test_discovery"]
        )
        names = [c.__name__ for c in classes if c.__name__ == "_DiscoveryAgent"]
        assert len(names) == 1

    def test_does_not_include_base_agent_class(self) -> None:
        classes = discover_agent_classes(["tests.test_ui.test_discovery"])
        assert Agent not in classes
