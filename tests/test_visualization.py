"""Tests for the visualization system improvements."""

from collections.abc import Generator
from typing import Any

import pytest
from statemachine import State

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.visualization.agent_visualization import AgentVisualization
from agentspype.visualization.listening_visualization import ListeningVisualization
from agentspype.visualization.state_machine_visualization import (
    StateMachineVisualization,
)

# === Shared mock classes ===


class VizStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    stop = idle.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def after_transition(self, event: str, state: State) -> None:
        pass


class VizListening(AgentListening):
    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class VizPublishing(StateAgentPublishing):
    def publish(self, event_publication: Any, event_data: Any) -> None:
        pass


class MockComponent:
    """A generic mock component with a name attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


class ComponentAgent(Agent):
    """An agent that exposes sub-components."""

    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=VizListening,
        state_machine_class=VizStateMachine,
        status_class=AgentStatus,
    )

    def initialize(self) -> None:
        self._components = [
            MockComponent("OrderComponent"),
            MockComponent("TransferComponent"),
        ]

    def get_components(self) -> list[Any]:
        return self._components


class PlainAgent(Agent):
    """An agent with no components (default get_components)."""

    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=VizListening,
        state_machine_class=VizStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def component_agent() -> Generator[ComponentAgent]:
    agent = ComponentAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


@pytest.fixture
def plain_agent() -> Generator[PlainAgent]:
    agent = PlainAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


# === 1. Component Composition Visualization Tests ===


class TestComponentVisualization:
    def test_agent_get_components_default_empty(self, plain_agent: PlainAgent) -> None:
        """Default get_components returns empty list."""
        assert plain_agent.get_components() == []

    def test_agent_get_components_returns_components(
        self, component_agent: ComponentAgent
    ) -> None:
        """Overridden get_components returns the agent's components."""
        components = component_agent.get_components()
        assert len(components) == 2
        assert components[0].name == "OrderComponent"
        assert components[1].name == "TransferComponent"

    def test_component_nodes_in_diagram(self, component_agent: ComponentAgent) -> None:
        """Component nodes appear in the agent's comprehensive diagram."""
        viz = AgentVisualization()
        graph = viz.create_visualization(component_agent)

        node_names = {node.get_name().strip('"') for node in graph.get_node_list()}
        assert "comp_0_OrderComponent" in node_names
        assert "comp_1_TransferComponent" in node_names

    def test_component_edges_in_diagram(self, component_agent: ComponentAgent) -> None:
        """Edges connect the agent node to each component node."""
        viz = AgentVisualization()
        graph = viz.create_visualization(component_agent)

        edge_pairs = [
            (e.get_source().strip('"'), e.get_destination().strip('"'))
            for e in graph.get_edge_list()
        ]
        assert ("ComponentAgent", "comp_0_OrderComponent") in edge_pairs
        assert ("ComponentAgent", "comp_1_TransferComponent") in edge_pairs

    def test_no_component_nodes_when_disabled(
        self, component_agent: ComponentAgent
    ) -> None:
        """When include_components=False, component nodes are absent."""
        viz = AgentVisualization()
        graph = viz.create_visualization(component_agent, include_components=False)

        node_names = {node.get_name().strip('"') for node in graph.get_node_list()}
        assert "comp_OrderComponent" not in node_names
        assert "comp_TransferComponent" not in node_names

    def test_no_component_nodes_for_plain_agent(self, plain_agent: PlainAgent) -> None:
        """An agent with no components produces no component nodes."""
        viz = AgentVisualization()
        graph = viz.create_visualization(plain_agent)

        node_names = {node.get_name().strip('"') for node in graph.get_node_list()}
        comp_nodes = [n for n in node_names if n.startswith("comp_")]
        assert len(comp_nodes) == 0

    def test_component_without_name_uses_class_name(
        self, plain_agent: PlainAgent
    ) -> None:
        """If a component lacks a 'name' attr, its class __name__ is used."""

        class Gadget:
            pass

        plain_agent.get_components = lambda: [Gadget()]  # type: ignore[assignment]
        viz = AgentVisualization()
        graph = viz.create_visualization(plain_agent)

        node_names = {node.get_name().strip('"') for node in graph.get_node_list()}
        assert "comp_0_Gadget" in node_names

    def test_components_with_same_name_produce_separate_nodes(
        self, plain_agent: PlainAgent
    ) -> None:
        """Two components with the same name appear as separate nodes thanks to indexed IDs."""
        plain_agent.get_components = lambda: [  # type: ignore[assignment]
            MockComponent("Duplicate"),
            MockComponent("Duplicate"),
        ]
        viz = AgentVisualization()
        graph = viz.create_visualization(plain_agent)

        node_names = [
            node.get_name().strip('"')
            for node in graph.get_node_list()
            if node.get_name().strip('"').startswith("comp_")
        ]
        assert "comp_0_Duplicate" in node_names
        assert "comp_1_Duplicate" in node_names
        assert len(node_names) == 2


# === 2. Publisher Deduplication Tests ===


class TestPublisherDeduplication:
    def _make_listening_class_with_defs(
        self, event_definitions: dict[str, Any]
    ) -> type:
        """Create a mock listening class with given event_definitions."""

        class FakeListening:
            @classmethod
            def get_event_definitions(cls) -> dict[str, Any]:
                return event_definitions

        FakeListening.__name__ = "TestListening"
        return FakeListening

    def test_single_publisher_single_node(self) -> None:
        """A publisher referenced once results in one node."""

        class PubA:
            pass

        defs = {
            "on_data": {"event_tag": None, "publisher_class": PubA},
        }
        listening_cls = self._make_listening_class_with_defs(defs)

        viz = ListeningVisualization()
        graph = viz.create_visualization(listening_cls)

        pub_nodes = [
            n for n in graph.get_node_list() if n.get_name().strip('"') == "pub_PubA"
        ]
        assert len(pub_nodes) == 1

    def test_same_publisher_multiple_subscriptions_one_node(self) -> None:
        """The same publisher referenced by two subscriptions produces one node."""

        class PubA:
            pass

        defs = {
            "on_data_1": {"event_tag": None, "publisher_class": PubA},
            "on_data_2": {"event_tag": None, "publisher_class": PubA},
        }
        listening_cls = self._make_listening_class_with_defs(defs)

        viz = ListeningVisualization()
        graph = viz.create_visualization(listening_cls)

        pub_nodes = [
            n for n in graph.get_node_list() if n.get_name().strip('"') == "pub_PubA"
        ]
        assert len(pub_nodes) == 1

    def test_same_publisher_multiple_edges(self) -> None:
        """Two subscriptions from the same publisher still produce two edges."""

        class PubA:
            pass

        defs = {
            "on_data_1": {"event_tag": None, "publisher_class": PubA},
            "on_data_2": {"event_tag": None, "publisher_class": PubA},
        }
        listening_cls = self._make_listening_class_with_defs(defs)

        viz = ListeningVisualization()
        graph = viz.create_visualization(listening_cls)

        pub_edges = [
            e for e in graph.get_edge_list() if e.get_source().strip('"') == "pub_PubA"
        ]
        assert len(pub_edges) == 2

    def test_different_publishers_separate_nodes(self) -> None:
        """Different publisher classes each get their own node."""

        class PubA:
            pass

        class PubB:
            pass

        defs = {
            "on_data_a": {"event_tag": None, "publisher_class": PubA},
            "on_data_b": {"event_tag": None, "publisher_class": PubB},
        }
        listening_cls = self._make_listening_class_with_defs(defs)

        viz = ListeningVisualization()
        graph = viz.create_visualization(listening_cls)

        pub_node_names = {
            n.get_name().strip('"')
            for n in graph.get_node_list()
            if n.get_name().strip('"').startswith("pub_")
        }
        assert pub_node_names == {"pub_PubA", "pub_PubB"}


# === 3. Custom Edge Styling Tests ===


class TestCustomEdgeStyling:
    def test_default_edge_styles_applied(self, plain_agent: PlainAgent) -> None:
        """Default edge style map colours start/stop edges."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(plain_agent.machine)

        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label is None:
                continue
            label = label.strip('"')
            if label == "start":
                assert edge.obj_dict["attributes"]["color"] == "green"
                assert edge.obj_dict["attributes"]["style"] == "dashed"
            elif label == "stop":
                assert edge.obj_dict["attributes"]["color"] == "red"
                assert edge.obj_dict["attributes"]["style"] == "dashed"

    def test_custom_edge_style_map_overrides_defaults(
        self, plain_agent: PlainAgent
    ) -> None:
        """A user-provided edge_style_map overrides the defaults."""
        custom_map = {
            "start": {"color": "forestgreen", "style": "bold"},
            "stop": {"color": "firebrick", "penwidth": "2"},
        }
        viz = StateMachineVisualization()
        graph = viz.create_visualization(plain_agent.machine, edge_style_map=custom_map)

        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label is None:
                continue
            label = label.strip('"')
            if label == "start":
                assert edge.obj_dict["attributes"]["color"] == "forestgreen"
                assert edge.obj_dict["attributes"]["style"] == "bold"
            elif label == "stop":
                assert edge.obj_dict["attributes"]["color"] == "firebrick"
                assert edge.obj_dict["attributes"]["penwidth"] == "2"

    def test_custom_edge_style_for_new_event(self, plain_agent: PlainAgent) -> None:
        """A custom entry for an event not in the defaults is applied."""
        # "start" and "stop" are actual transitions in VizStateMachine
        # We add a custom style that targets "start" with a unique colour
        custom_map = {
            "start": {"color": "purple"},
        }
        viz = StateMachineVisualization()
        graph = viz.create_visualization(plain_agent.machine, edge_style_map=custom_map)

        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label is None:
                continue
            label = label.strip('"')
            if label == "start":
                assert edge.obj_dict["attributes"]["color"] == "purple"

    def test_edge_style_map_passed_through_agent_visualize(
        self, plain_agent: PlainAgent
    ) -> None:
        """edge_style_map passes through Agent.visualize to state machine viz."""
        custom_map = {"start": {"color": "cyan"}}
        graph = plain_agent.visualize(edge_style_map=custom_map)

        start_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and e.get_label().strip('"') == "start"
        ]
        assert len(start_edges) > 0
        for edge in start_edges:
            assert edge.obj_dict["attributes"]["color"] == "cyan"

    def test_default_class_attribute(self) -> None:
        """The DEFAULT_EDGE_STYLE_MAP class attribute is accessible."""
        assert "start" in StateMachineVisualization.DEFAULT_EDGE_STYLE_MAP
        assert "stop" in StateMachineVisualization.DEFAULT_EDGE_STYLE_MAP
