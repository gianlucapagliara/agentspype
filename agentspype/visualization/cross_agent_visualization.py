"""Cross-agent relationship visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization
from .theme import Theme

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent


class CrossAgentVisualization(BaseVisualization):
    """Visualization showing relationships between multiple agents.

    Renders a diagram with:
    - Agent nodes for each agent
    - Event wiring edges (publisher agent -> listener agent)
    - Parent-child hierarchy edges (instance-level only)
    - External publisher nodes for non-agent event sources
    """

    _COLORS = {
        "agent_fill": Theme.CROSS_AGENT_FILL,
        "agent_border": Theme.CROSS_AGENT_BORDER,
        "event_edge": Theme.CROSS_EVENT_EDGE,
        "event_label": Theme.CROSS_EVENT_LABEL,
        "parent_child_edge": Theme.CROSS_PARENT_CHILD_EDGE,
        "external_fill": Theme.CROSS_EXTERNAL_FILL,
        "external_border": Theme.CROSS_EXTERNAL_BORDER,
        "self_edge": Theme.CROSS_SELF_EDGE,
    }

    def create_visualization(
        self,
        target: Any,
        graph: pydot.Dot | None = None,
        show_event_wiring: bool = True,
        show_parent_child: bool = True,
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create a cross-agent relationship diagram.

        Args:
            target: A list of Agent instances or Agent classes.
            graph: An existing graph to add to, or None to create a new one.
            show_event_wiring: Whether to show event publishing/listening edges.
            show_parent_child: Whether to show parent-child hierarchy edges
                (only works with agent instances, not classes).
            **kwargs: Additional arguments.

        Returns:
            pydot.Dot: The generated cross-agent diagram.
        """
        agents = target
        if not agents:
            if graph is None:
                graph = self._create_base_graph()
            return graph

        # Detect whether we have instances or classes
        is_instances = not isinstance(agents[0], type)

        if graph is None:
            graph = self._create_base_graph()

        # Build agent nodes
        agent_classes = self._add_agent_nodes(graph, agents, is_instances)

        # Event wiring
        if show_event_wiring:
            if is_instances:
                self._add_event_wiring_from_instances(graph, agents, agent_classes)
            else:
                self._add_event_wiring_from_classes(graph, agents, agent_classes)

        # Parent-child hierarchy (instance-level only)
        if show_parent_child and is_instances:
            self._add_parent_child_edges(graph, agents)

        return graph

    def _create_base_graph(self) -> pydot.Dot:
        """Create the base graph with cross-agent styling."""
        dot_graph = pydot.Dot(graph_type="digraph")
        dot_graph.obj_dict["attributes"]["rankdir"] = self.GRAPH_RANKDIR
        dot_graph.obj_dict["attributes"]["fontname"] = self.FONT_NAME
        dot_graph.obj_dict["attributes"]["fontsize"] = self.FONT_SIZE
        dot_graph.obj_dict["attributes"]["bgcolor"] = "transparent"
        dot_graph.obj_dict["attributes"]["pad"] = "0.5"
        dot_graph.obj_dict["attributes"]["nodesep"] = "1.0"
        dot_graph.obj_dict["attributes"]["ranksep"] = "1.5"
        return dot_graph

    def _get_agent_class(self, agent: Any) -> type:
        """Get the agent class from an instance or class."""
        return agent if isinstance(agent, type) else type(agent)

    def _get_agent_node_id(self, agent: Any) -> str:
        """Get a unique node ID for an agent."""
        cls = self._get_agent_class(agent)
        return f"agent_{cls.__name__}"

    def _add_agent_nodes(
        self,
        graph: pydot.Dot,
        agents: list[Any],
        is_instances: bool,
    ) -> dict[str, type]:
        """Add agent nodes and return a map of node_id -> agent_class."""
        agent_classes: dict[str, type] = {}

        for agent in agents:
            cls = self._get_agent_class(agent)
            node_id = self._get_agent_node_id(agent)

            if node_id in agent_classes:
                continue

            agent_classes[node_id] = cls

            # Build label with class name and optional state info
            label = cls.__name__
            if is_instances and hasattr(agent, "machine"):
                current = agent.machine.current_state
                label = f"{{{cls.__name__} | {current.name or current.id}}}"

            node = pydot.Node(
                node_id,
                label=label,
                shape="Mrecord",
                style="filled",
                fillcolor=self._COLORS["agent_fill"],
                color=self._COLORS["agent_border"],
                fontname=self.FONT_NAME,
                fontsize=self.FONT_SIZE,
                penwidth="2",
            )
            graph.add_node(node)

        return agent_classes

    def _build_publishing_class_to_agent_map(
        self, agents: list[Any]
    ) -> dict[type, str]:
        """Build a map from publishing class -> agent node_id."""
        pub_to_agent: dict[type, str] = {}

        for agent in agents:
            cls = self._get_agent_class(agent)
            if hasattr(cls, "definition"):
                pub_cls = cls.definition.events_publishing_class
                pub_to_agent[pub_cls] = self._get_agent_node_id(agent)
                # Also map parent classes in MRO for subclass matching
                for base in pub_cls.__mro__:
                    if base not in pub_to_agent:
                        pub_to_agent[base] = self._get_agent_node_id(agent)

        return pub_to_agent

    def _add_event_wiring_from_classes(
        self,
        graph: pydot.Dot,
        agents: list[Any],
        agent_classes: dict[str, type],
    ) -> None:
        """Add event wiring edges using class-level metadata."""
        pub_to_agent = self._build_publishing_class_to_agent_map(agents)
        seen_edges: set[tuple[str, str, str]] = set()
        external_nodes: set[str] = set()

        for agent in agents:
            cls = self._get_agent_class(agent)
            if not hasattr(cls, "definition"):
                continue

            listener_node_id = self._get_agent_node_id(agent)
            listening_cls = cls.definition.events_listening_class

            if not hasattr(listening_cls, "get_event_definitions"):
                continue

            event_defs = listening_cls.get_event_definitions()
            for sub_name, subscription in event_defs.items():
                publisher_class = self._get_publisher_class(subscription)
                event_tag = self._get_event_tag(subscription)

                if publisher_class is None:
                    continue

                tag_label = self._format_event_tag(event_tag) if event_tag else sub_name

                # Find the agent that owns this publisher class
                publisher_node_id = pub_to_agent.get(publisher_class)

                if publisher_node_id is None:
                    # External publisher — not owned by any agent
                    publisher_node_id = self._add_external_publisher_node(
                        graph, publisher_class, external_nodes
                    )

                edge_key = (publisher_node_id, listener_node_id, tag_label)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                self._add_event_edge(
                    graph, publisher_node_id, listener_node_id, tag_label
                )

    def _add_event_wiring_from_instances(
        self,
        graph: pydot.Dot,
        agents: list["Agent"],
        agent_classes: dict[str, type],
    ) -> None:
        """Add event wiring edges using instance-level data."""
        # Reuse class-level logic — instances have the same class metadata
        self._add_event_wiring_from_classes(graph, agents, agent_classes)

    def _get_publisher_class(self, subscription: Any) -> type | None:
        """Extract publisher class from a subscription (object or dict)."""
        if isinstance(subscription, dict):
            return subscription.get("publisher_class")
        return getattr(subscription, "publisher_class", None)

    def _get_event_tag(self, subscription: Any) -> Any:
        """Extract original event tag (enum member) from a subscription."""
        if isinstance(subscription, dict):
            return subscription.get("event_tag")
        return getattr(subscription, "original_tag", None) or getattr(
            subscription, "event_tag", None
        )

    def _format_event_tag(self, event_tag: Any) -> str:
        """Format an event tag for display."""
        if isinstance(event_tag, list):
            return ", ".join(self._format_single_tag(t) for t in event_tag)
        return self._format_single_tag(event_tag)

    def _format_single_tag(self, tag: Any) -> str:
        """Format a single event tag value."""
        if hasattr(tag, "name"):
            return str(tag.name)
        return str(tag)

    def _add_external_publisher_node(
        self,
        graph: pydot.Dot,
        publisher_class: type,
        external_nodes: set[str],
    ) -> str:
        """Add a node for an external (non-agent) publisher."""
        name = (
            publisher_class.__name__
            if hasattr(publisher_class, "__name__")
            else str(publisher_class)
        )
        node_id = f"ext_{name}"

        if node_id not in external_nodes:
            external_nodes.add(node_id)
            node = pydot.Node(
                node_id,
                label=name,
                shape="rectangle",
                style="filled, dashed",
                fillcolor=self._COLORS["external_fill"],
                color=self._COLORS["external_border"],
                fontname=self.FONT_NAME,
                fontsize=self.FONT_SIZE,
            )
            graph.add_node(node)

        return node_id

    def _add_event_edge(
        self,
        graph: pydot.Dot,
        source_id: str,
        target_id: str,
        label: str,
    ) -> None:
        """Add an event wiring edge."""
        is_self = source_id == target_id
        color = self._COLORS["self_edge"] if is_self else self._COLORS["event_edge"]

        edge = pydot.Edge(
            source_id,
            target_id,
            label=label,
            color=color,
            fontcolor=self._COLORS["event_label"],
            fontname=self.FONT_NAME,
            fontsize=self.FONT_SIZE,
            style="solid",
            arrowhead="vee",
            penwidth="1.5",
        )
        graph.add_edge(edge)

    def _add_parent_child_edges(
        self,
        graph: pydot.Dot,
        agents: list["Agent"],
    ) -> None:
        """Add parent-child hierarchy edges between agent instances."""
        agent_id_map: dict[int, Agent] = {id(a): a for a in agents}

        for agent in agents:
            if agent.parent_id is not None and agent.parent_id in agent_id_map:
                parent = agent_id_map[agent.parent_id]
                parent_node_id = self._get_agent_node_id(parent)
                child_node_id = self._get_agent_node_id(agent)

                edge = pydot.Edge(
                    parent_node_id,
                    child_node_id,
                    label="parent",
                    color=self._COLORS["parent_child_edge"],
                    fontcolor=self._COLORS["parent_child_edge"],
                    fontname=self.FONT_NAME,
                    fontsize=self.FONT_SIZE,
                    style="dashed",
                    arrowhead="vee",
                    penwidth="1.5",
                )
                graph.add_edge(edge)
