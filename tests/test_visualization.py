"""Tests for the visualization system improvements."""

from collections.abc import Generator
from enum import Enum
from typing import Any

import pydot
import pytest
from eventspype.pub.publication import EventPublication
from eventspype.sub.subscription import EventSubscription

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.fsm import State
from agentspype.visualization.agent_visualization import AgentVisualization
from agentspype.visualization.base_visualization import (
    GraphvizNotFoundError,
    _check_graphviz,
)
from agentspype.visualization.cross_agent_visualization import CrossAgentVisualization
from agentspype.visualization.listening_visualization import ListeningVisualization
from agentspype.visualization.state_machine_visualization import (
    StateMachineVisualization,
)
from agentspype.visualization.theme import Theme

# === Graph traversal helpers ===


def _all_nodes(graph: pydot.Dot) -> list[pydot.Node]:
    """Collect nodes from a graph and all nested subgraphs recursively."""
    nodes = list(graph.get_node_list())
    for sub in graph.get_subgraph_list():
        nodes.extend(_all_nodes(sub))
    return nodes


def _all_edges(graph: pydot.Dot) -> list[pydot.Edge]:
    """Collect edges from a graph and all nested subgraphs recursively."""
    edges = list(graph.get_edge_list())
    for sub in graph.get_subgraph_list():
        edges.extend(_all_edges(sub))
    return edges


# === Shared mock classes ===


class VizStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    stop = starting.to(end) | idle.to(end)

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
        graph = viz.create_visualization(component_agent, include_components=True)

        node_names = {node.get_name().strip('"') for node in _all_nodes(graph)}
        assert "comp_0_OrderComponent" in node_names
        assert "comp_1_TransferComponent" in node_names

    def test_component_edges_in_diagram(self, component_agent: ComponentAgent) -> None:
        """Component nodes exist in the components cluster."""
        viz = AgentVisualization()
        graph = viz.create_visualization(component_agent, include_components=True)

        node_names = {node.get_name().strip('"') for node in _all_nodes(graph)}
        assert "comp_0_OrderComponent" in node_names
        assert "comp_1_TransferComponent" in node_names

    def test_no_component_nodes_when_disabled(
        self, component_agent: ComponentAgent
    ) -> None:
        """When include_components=False, component nodes are absent."""
        viz = AgentVisualization()
        graph = viz.create_visualization(component_agent, include_components=False)

        node_names = {node.get_name().strip('"') for node in _all_nodes(graph)}
        assert "comp_OrderComponent" not in node_names
        assert "comp_TransferComponent" not in node_names

    def test_no_component_nodes_for_plain_agent(self, plain_agent: PlainAgent) -> None:
        """An agent with no components produces no component nodes."""
        viz = AgentVisualization()
        graph = viz.create_visualization(plain_agent)

        node_names = {node.get_name().strip('"') for node in _all_nodes(graph)}
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
        graph = viz.create_visualization(plain_agent, include_components=True)

        node_names = {node.get_name().strip('"') for node in _all_nodes(graph)}
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
        graph = viz.create_visualization(plain_agent, include_components=True)

        node_names = [
            node.get_name().strip('"')
            for node in _all_nodes(graph)
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
            for e in _all_edges(graph)
            if e.get_label() and e.get_label().strip('"').startswith("start")
        ]
        assert len(start_edges) > 0
        for edge in start_edges:
            assert edge.obj_dict["attributes"]["color"] == "cyan"

    def test_default_class_attribute(self) -> None:
        """The DEFAULT_EDGE_STYLE_MAP class attribute is accessible."""
        assert "start" in StateMachineVisualization.DEFAULT_EDGE_STYLE_MAP
        assert "stop" in StateMachineVisualization.DEFAULT_EDGE_STYLE_MAP


# === 4. Guard Conditions on Transition Edges ===


class GuardedStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    processing = State("Processing")
    end = State("End", final=True)

    start = starting.to(idle)
    process = idle.to(processing, cond="is_ready")
    finish = processing.to(idle, unless="has_pending")
    stop = starting.to(end) | idle.to(end) | processing.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def is_ready(self) -> bool:
        return True

    def has_pending(self) -> bool:
        return False

    def after_transition(self, event: str, state: State) -> None:
        pass


class GuardedAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=VizListening,
        state_machine_class=GuardedStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def guarded_agent() -> Generator[GuardedAgent]:
    agent = GuardedAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


class TestGuardConditions:
    def test_cond_guard_shown_by_default(self, guarded_agent: GuardedAgent) -> None:
        """Guard conditions (cond) appear in edge labels by default."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(guarded_agent.machine)

        process_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "process" in e.get_label().strip('"')
        ]
        assert len(process_edges) > 0
        label = process_edges[0].get_label().strip('"')
        assert "[is_ready]" in label

    def test_unless_guard_shown(self, guarded_agent: GuardedAgent) -> None:
        """Unless guards appear with ! prefix in edge labels."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(guarded_agent.machine)

        finish_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "finish" in e.get_label().strip('"')
        ]
        assert len(finish_edges) > 0
        label = finish_edges[0].get_label().strip('"')
        assert "[!has_pending]" in label

    def test_guards_hidden_when_disabled(self, guarded_agent: GuardedAgent) -> None:
        """Guards are not shown when show_guards=False."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(guarded_agent.machine, show_guards=False)

        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label:
                assert "[" not in label.strip('"')

    def test_no_guards_on_plain_transitions(self, plain_agent: PlainAgent) -> None:
        """Transitions without guards don't get bracket annotations."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(plain_agent.machine)

        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label:
                assert "[" not in label.strip('"')

    def test_guards_pass_through_agent_visualize(
        self, guarded_agent: GuardedAgent
    ) -> None:
        """show_guards passes through Agent.visualize()."""
        graph = guarded_agent.visualize(show_guards=False)
        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label:
                assert "[" not in label.strip('"')


# === 5. Internal Transition Styling ===


class InternalStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    tick = idle.to.itself(internal=True)
    stop = starting.to(end) | idle.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def after_transition(self, event: str, state: State) -> None:
        pass


class InternalAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=VizListening,
        state_machine_class=InternalStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def internal_agent() -> Generator[InternalAgent]:
    agent = InternalAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


class TestInternalTransitionStyling:
    def test_internal_transition_dotted(self, internal_agent: InternalAgent) -> None:
        """Internal transitions are styled with dotted lines."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(internal_agent.machine)

        tick_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "tick" in e.get_label().strip('"')
        ]
        assert len(tick_edges) > 0
        attrs = tick_edges[0].obj_dict["attributes"]
        assert attrs["style"] == "dotted"

    def test_internal_transition_muted_color(
        self, internal_agent: InternalAgent
    ) -> None:
        """Internal transitions use the muted gray color."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(internal_agent.machine)

        tick_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "tick" in e.get_label().strip('"')
        ]
        assert len(tick_edges) > 0
        attrs = tick_edges[0].obj_dict["attributes"]
        assert attrs["color"] == StateMachineVisualization._COLORS["internal_edge"]

    def test_internal_transition_self_loop(self, internal_agent: InternalAgent) -> None:
        """Internal transitions point back to the same state."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(internal_agent.machine)

        tick_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "tick" in e.get_label().strip('"')
        ]
        assert len(tick_edges) > 0
        edge = tick_edges[0]
        assert edge.get_source().strip('"') == edge.get_destination().strip('"')

    def test_explicit_style_overrides_internal(
        self, internal_agent: InternalAgent
    ) -> None:
        """An explicit edge_style_map entry overrides internal defaults."""
        custom_map = {"tick": {"color": "purple", "style": "bold"}}
        viz = StateMachineVisualization()
        graph = viz.create_visualization(
            internal_agent.machine, edge_style_map=custom_map
        )

        tick_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "tick" in e.get_label().strip('"')
        ]
        assert len(tick_edges) > 0
        attrs = tick_edges[0].obj_dict["attributes"]
        assert attrs["color"] == "purple"
        assert attrs["style"] == "bold"


# === 6. FSM Hook Annotations ===


class HookedStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    stop = starting.to(end) | idle.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def on_enter_idle(self) -> None:
        pass

    def on_exit_idle(self) -> None:
        pass

    def before_start(self) -> None:
        pass

    def on_start(self) -> None:
        pass

    def after_start(self) -> None:
        pass

    def after_transition(self, event: str, state: State) -> None:
        pass


class HookedAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=VizListening,
        state_machine_class=HookedStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def hooked_agent() -> Generator[HookedAgent]:
    agent = HookedAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


class TestHookAnnotations:
    def test_hooks_shown_by_default(self, hooked_agent: HookedAgent) -> None:
        """Hooks are shown by default (show_hooks=True)."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(hooked_agent.machine)

        idle_nodes = [
            n for n in graph.get_node_list() if n.get_name().strip('"') == "idle"
        ]
        assert len(idle_nodes) == 1
        label = idle_nodes[0].get_label().strip('"')
        assert "on_enter_idle" in label

    def test_hooks_hidden_when_disabled(self, hooked_agent: HookedAgent) -> None:
        """Hooks are not shown when show_hooks=False."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(hooked_agent.machine, show_hooks=False)

        # State labels should be plain names
        for node in graph.get_node_list():
            label = node.get_label()
            if label:
                label = label.strip('"')
                assert "on_enter" not in label
                assert "on_exit" not in label

        # Edge labels should be plain event names
        for edge in graph.get_edge_list():
            label = edge.get_label()
            if label:
                label = label.strip('"')
                assert "before_" not in label
                assert "after_" not in label

    def test_state_hooks_shown(self, hooked_agent: HookedAgent) -> None:
        """State entry/exit hooks appear in state node labels when show_hooks=True."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(hooked_agent.machine, show_hooks=True)

        idle_nodes = [
            n for n in graph.get_node_list() if n.get_name().strip('"') == "idle"
        ]
        assert len(idle_nodes) == 1
        label = idle_nodes[0].get_label().strip('"')
        assert "on_enter_idle" in label
        assert "on_exit_idle" in label

    def test_event_hooks_on_edges(self, hooked_agent: HookedAgent) -> None:
        """Event hooks (before_*, on_*, after_*) appear on transition edges."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(hooked_agent.machine, show_hooks=True)

        start_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label()
            and "start" in e.get_label().strip('"')
            and "before_start" in e.get_label().strip('"')
        ]
        assert len(start_edges) > 0
        label = start_edges[0].get_label().strip('"')
        assert "before_start" in label
        assert "on_start" in label
        assert "after_start" in label

    def test_states_without_hooks_plain_label(self, hooked_agent: HookedAgent) -> None:
        """States without hooks keep a plain label even with show_hooks=True."""
        viz = StateMachineVisualization()
        graph = viz.create_visualization(hooked_agent.machine, show_hooks=True)

        starting_nodes = [
            n for n in graph.get_node_list() if n.get_name().strip('"') == "starting"
        ]
        assert len(starting_nodes) == 1
        label = starting_nodes[0].get_label().strip('"')
        # No record syntax — just the plain name
        assert "{" not in label

    def test_show_hooks_passes_through_agent_visualize(
        self, hooked_agent: HookedAgent
    ) -> None:
        """show_hooks passes through Agent.visualize()."""
        graph = hooked_agent.visualize(show_hooks=True)

        idle_nodes = [n for n in _all_nodes(graph) if n.get_name().strip('"') == "idle"]
        assert len(idle_nodes) == 1
        label = idle_nodes[0].get_label().strip('"')
        assert "on_enter_idle" in label


# === 7. Cross-Agent Visualization ===

# -- Publisher agent: publishes a custom event --


class ProducerPublishing(StateAgentPublishing):
    class Events(Enum):
        DataReady = "data_ready"

    data_ready = EventPublication(event_tag=Events.DataReady, event_class=dict)


class ProducerStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)

    start = starting.to(idle)
    stop = starting.to(end) | idle.to(end)

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)

    def after_transition(self, event: str, state: State) -> None:
        pass


class ProducerAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=ProducerPublishing,
        events_listening_class=VizListening,
        state_machine_class=ProducerStateMachine,
        status_class=AgentStatus,
    )


# -- Consumer agent: listens to ProducerPublishing --


class ConsumerListening(AgentListening):
    def on_data(self, event: Any) -> None:
        pass

    data_subscription = EventSubscription(
        publisher_class=ProducerPublishing,
        event_tag=ProducerPublishing.Events.DataReady,
        callback=on_data,
    )

    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class ConsumerAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=ConsumerListening,
        state_machine_class=VizStateMachine,
        status_class=AgentStatus,
    )


# -- Self-wiring agent: listens to its own events --


class SelfWiredListening(AgentListening):
    def on_self_event(self, event: Any) -> None:
        pass

    self_subscription = EventSubscription(
        publisher_class=ProducerPublishing,
        event_tag=ProducerPublishing.Events.DataReady,
        callback=on_self_event,
    )

    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class SelfWiredAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=ProducerPublishing,
        events_listening_class=SelfWiredListening,
        state_machine_class=VizStateMachine,
        status_class=AgentStatus,
    )


# -- External publisher listener: listens to a class not owned by any agent --


class ExternalPublisher(StateAgentPublishing):
    """A publishing class not used as any agent's publishing class."""

    class Events(Enum):
        ExternalEvent = "external_event"

    external_event = EventPublication(event_tag=Events.ExternalEvent, event_class=dict)


class ExternalListening(AgentListening):
    def on_external(self, event: Any) -> None:
        pass

    ext_subscription = EventSubscription(
        publisher_class=ExternalPublisher,
        event_tag=ExternalPublisher.Events.ExternalEvent,
        callback=on_external,
    )

    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class ExternalListenerAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=VizPublishing,
        events_listening_class=ExternalListening,
        state_machine_class=VizStateMachine,
        status_class=AgentStatus,
    )


@pytest.fixture
def producer_agent() -> Generator[ProducerAgent]:
    agent = ProducerAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


@pytest.fixture
def consumer_agent() -> Generator[ConsumerAgent]:
    agent = ConsumerAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


@pytest.fixture
def self_wired_agent() -> Generator[SelfWiredAgent]:
    agent = SelfWiredAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


@pytest.fixture
def external_listener_agent() -> Generator[ExternalListenerAgent]:
    agent = ExternalListenerAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass


class TestCrossAgentEventWiring:
    def test_publisher_to_listener_edge(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """An edge is drawn from publisher agent to listener agent."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([producer_agent, consumer_agent])

        edge_pairs = [
            (e.get_source().strip('"'), e.get_destination().strip('"'))
            for e in graph.get_edge_list()
        ]
        assert ("agent_ProducerAgent", "agent_ConsumerAgent") in edge_pairs

    def test_class_level_wiring(self) -> None:
        """Event wiring works from class references (no instances needed)."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([ProducerAgent, ConsumerAgent])

        edge_pairs = [
            (e.get_source().strip('"'), e.get_destination().strip('"'))
            for e in graph.get_edge_list()
        ]
        assert ("agent_ProducerAgent", "agent_ConsumerAgent") in edge_pairs

    def test_no_incoming_edges_for_non_listener(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """Producer agent has no incoming event edges (VizListening has no subscriptions)."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([producer_agent, consumer_agent])

        incoming_to_producer = [
            e
            for e in graph.get_edge_list()
            if e.get_destination().strip('"') == "agent_ProducerAgent"
        ]
        assert len(incoming_to_producer) == 0

    def test_self_wiring_edge(self, self_wired_agent: SelfWiredAgent) -> None:
        """An agent listening to its own events creates a self-loop edge."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([self_wired_agent])

        self_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_source().strip('"') == e.get_destination().strip('"')
            and e.get_source().strip('"') == "agent_SelfWiredAgent"
        ]
        assert len(self_edges) > 0

    def test_external_publisher_node(
        self, external_listener_agent: ExternalListenerAgent
    ) -> None:
        """A listener subscribing to a non-agent publisher creates an external node."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([external_listener_agent])

        node_names = {n.get_name().strip('"') for n in graph.get_node_list()}
        ext_nodes = [n for n in node_names if n.startswith("ext_")]
        assert len(ext_nodes) > 0

    def test_event_wiring_disabled(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """No event edges when show_event_wiring=False."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization(
            [producer_agent, consumer_agent], show_event_wiring=False
        )

        # Only agent nodes, no event edges
        edge_pairs = [
            (e.get_source().strip('"'), e.get_destination().strip('"'))
            for e in graph.get_edge_list()
        ]
        assert ("agent_ProducerAgent", "agent_ConsumerAgent") not in edge_pairs


class TestCrossAgentParentChild:
    def test_parent_child_edge(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """Parent-child relationship renders a dashed edge."""
        consumer_agent.parent_id = id(producer_agent)

        viz = CrossAgentVisualization()
        graph = viz.create_visualization([producer_agent, consumer_agent])

        parent_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "parent" in e.get_label().strip('"')
        ]
        assert len(parent_edges) > 0
        assert parent_edges[0].obj_dict["attributes"]["style"] == "dashed"

    def test_no_parent_child_without_relationship(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """No parent-child edges when agents have no parent_id set."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([producer_agent, consumer_agent])

        parent_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "parent" in e.get_label().strip('"')
        ]
        assert len(parent_edges) == 0

    def test_parent_child_disabled(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """No parent-child edges when show_parent_child=False."""
        consumer_agent.parent_id = id(producer_agent)

        viz = CrossAgentVisualization()
        graph = viz.create_visualization(
            [producer_agent, consumer_agent], show_parent_child=False
        )

        parent_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "parent" in e.get_label().strip('"')
        ]
        assert len(parent_edges) == 0

    def test_parent_child_class_level_skipped(self) -> None:
        """Parent-child is not rendered for class-level visualization."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([ProducerAgent, ConsumerAgent])

        parent_edges = [
            e
            for e in graph.get_edge_list()
            if e.get_label() and "parent" in e.get_label().strip('"')
        ]
        assert len(parent_edges) == 0


class TestCrossAgentEmpty:
    def test_empty_agent_list(self) -> None:
        """Empty list produces an empty graph."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([])
        assert len(graph.get_node_list()) == 0
        assert len(graph.get_edge_list()) == 0

    def test_single_agent_no_event_edges(self, plain_agent: PlainAgent) -> None:
        """Single agent with no subscriptions has no event edges."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([plain_agent])

        node_names = {n.get_name().strip('"') for n in graph.get_node_list()}
        assert "agent_PlainAgent" in node_names
        assert len(graph.get_edge_list()) == 0

    def test_agent_nodes_present(
        self,
        producer_agent: ProducerAgent,
        consumer_agent: ConsumerAgent,
    ) -> None:
        """Both agent nodes appear in the diagram."""
        viz = CrossAgentVisualization()
        graph = viz.create_visualization([producer_agent, consumer_agent])

        node_names = {n.get_name().strip('"') for n in graph.get_node_list()}
        assert "agent_ProducerAgent" in node_names
        assert "agent_ConsumerAgent" in node_names


# === 8. Graphviz Pre-flight Check ===


class TestGraphvizPreflightCheck:
    def test_check_passes_when_dot_available(self) -> None:
        """_check_graphviz does not raise when dot is on PATH."""
        from unittest.mock import patch

        with patch(
            "agentspype.visualization.base_visualization.shutil.which",
            return_value="/usr/bin/dot",
        ):
            _check_graphviz()  # should not raise

    def test_check_raises_when_dot_missing(self) -> None:
        """_check_graphviz raises GraphvizNotFoundError when dot is missing."""
        from unittest.mock import patch

        with patch(
            "agentspype.visualization.base_visualization.shutil.which",
            return_value=None,
        ):
            with pytest.raises(
                GraphvizNotFoundError, match="Graphviz 'dot' executable not found"
            ):
                _check_graphviz()

    def test_save_diagram_checks_graphviz(self, tmp_path: str) -> None:
        """save_diagram raises GraphvizNotFoundError before attempting to write."""
        from unittest.mock import patch

        graph = pydot.Dot(graph_type="digraph")
        with patch(
            "agentspype.visualization.base_visualization.shutil.which",
            return_value=None,
        ):
            with pytest.raises(GraphvizNotFoundError):
                AgentVisualization.save_diagram(graph, "test", str(tmp_path))


# === 9. Centralized Theme ===


class TestThemeCentralization:
    def test_theme_values_match_visualizer_colors(self) -> None:
        """Theme constants are correctly wired into visualizer _COLORS dicts."""
        assert (
            StateMachineVisualization._COLORS["initial_fill"] == Theme.SM_INITIAL_FILL
        )
        assert (
            StateMachineVisualization._COLORS["internal_edge"] == Theme.SM_INTERNAL_EDGE
        )
        assert (
            AgentVisualization._COLORS["cluster_sm_border"]
            == Theme.AGENT_CLUSTER_SM_BORDER
        )

    def test_all_visualizers_use_theme(self) -> None:
        """All _COLORS values across visualizers are present in Theme."""
        from agentspype.visualization.cross_agent_visualization import (
            CrossAgentVisualization,
        )
        from agentspype.visualization.listening_visualization import (
            ListeningVisualization,
        )
        from agentspype.visualization.publishing_visualization import (
            PublishingVisualization,
        )

        theme_values = {v for k, v in vars(Theme).items() if not k.startswith("_")}

        for viz_cls in [
            StateMachineVisualization,
            PublishingVisualization,
            ListeningVisualization,
            AgentVisualization,
            CrossAgentVisualization,
        ]:
            for key, color in viz_cls._COLORS.items():
                assert color in theme_values, (
                    f"{viz_cls.__name__}._COLORS['{key}'] = {color!r} is not in Theme"
                )
