"""Comprehensive agent visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization
from .listening_visualization import ListeningVisualization
from .publishing_visualization import PublishingVisualization
from .state_machine_visualization import StateMachineVisualization
from .theme import Theme

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent

# Section definitions: (key, label, color_border_key, color_bg_key)
_SECTIONS = [
    ("state_machine", "State Machine", "cluster_sm_border", "cluster_sm_bg"),
    ("publishing", "Publishing", "cluster_pub_border", "cluster_pub_bg"),
    ("listening", "Listening", "cluster_listen_border", "cluster_listen_bg"),
    ("components", "Components", "cluster_comp_border", "cluster_comp_bg"),
]


class AgentVisualization(BaseVisualization):
    """Comprehensive visualization for agents.

    Renders a hub node with the agent name and labeled edges into each
    section cluster (state machine, publishing, listening, components).
    """

    _COLORS = {
        "component_fill": Theme.AGENT_COMPONENT_FILL,
        "component_border": Theme.AGENT_COMPONENT_BORDER,
        "cluster_sm_border": Theme.AGENT_CLUSTER_SM_BORDER,
        "cluster_sm_bg": Theme.AGENT_CLUSTER_SM_BG,
        "cluster_pub_border": Theme.AGENT_CLUSTER_PUB_BORDER,
        "cluster_pub_bg": Theme.AGENT_CLUSTER_PUB_BG,
        "cluster_listen_border": Theme.AGENT_CLUSTER_LISTEN_BORDER,
        "cluster_listen_bg": Theme.AGENT_CLUSTER_LISTEN_BG,
        "cluster_comp_border": Theme.AGENT_CLUSTER_COMP_BORDER,
        "cluster_comp_bg": Theme.AGENT_CLUSTER_COMP_BG,
        "cluster_label": Theme.AGENT_CLUSTER_LABEL,
    }

    _HUB_NODE_ID = "_agent_hub"

    def __init__(self) -> None:
        """Initialize the agent visualization with component visualizers."""
        self.state_machine_viz = StateMachineVisualization()
        self.publishing_viz = PublishingVisualization()
        self.listening_viz = ListeningVisualization()

    def create_visualization(
        self,
        target: Any,
        graph: pydot.Dot | None = None,
        include_state_machine: bool = True,
        include_publishing: bool = True,
        include_listening: bool = True,
        include_components: bool = False,
        show_current_state: bool = True,
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create a comprehensive visualization of the agent.

        A hub node with the agent class name sits on the left.  Labeled
        edges fan out to section clusters (state machine, publishing,
        listening, components) using ``compound=true`` and ``lhead``.

        Args:
            target: The agent instance to visualize.
            graph: An existing graph to add to, or ``None`` to create one.
            include_state_machine: Include the state-machine section.
            include_publishing: Include the publishing section.
            include_listening: Include the listening section.
            include_components: Include agent sub-components.
            show_current_state: Highlight the current state.
            **kwargs: Forwarded to sub-visualizers
                (e.g. ``edge_style_map`` for the state machine).

        Returns:
            The generated agent diagram.
        """
        agent = target
        agent_name = agent.__class__.__name__

        if graph is None:
            graph = self.create_graph(f"{agent_name}_agent")

        graph.obj_dict["attributes"]["rankdir"] = "LR"
        graph.obj_dict["attributes"]["compound"] = "true"
        graph.obj_dict["attributes"]["splines"] = "ortho"

        # Hub node
        graph.add_node(
            pydot.Node(
                self._HUB_NODE_ID,
                label=agent_name,
                shape="rectangle",
                style="filled,bold,rounded",
                fillcolor=Theme.SOFT_BLUE_FILL,
                color=Theme.SOFT_BLUE_BORDER,
                fontname=self.FONT_NAME,
                fontsize="14",
                penwidth="2.5",
            )
        )

        # --- State Machine ---
        if include_state_machine:
            current_state = agent.machine.current_state if show_current_state else None
            sm_graph = self.state_machine_viz.create_visualization(
                agent.machine, current_state=current_state, **kwargs
            )
            cluster = self._wrap_in_cluster(
                "state_machine",
                "State Machine",
                sm_graph,
                border_color=self._COLORS["cluster_sm_border"],
                bg_color=self._COLORS["cluster_sm_bg"],
            )
            graph.add_subgraph(cluster)
            first = self._first_node_id(sm_graph)
            if first:
                self._add_hub_edge(graph, first, "cluster_state_machine")

        # --- Publishing ---
        if include_publishing:
            pub_graph = self.publishing_viz.create_visualization(
                type(agent.publishing), **kwargs
            )
            cluster = self._wrap_in_cluster(
                "publishing",
                "Publishing",
                pub_graph,
                border_color=self._COLORS["cluster_pub_border"],
                bg_color=self._COLORS["cluster_pub_bg"],
            )
            graph.add_subgraph(cluster)
            first = self._first_node_id(pub_graph)
            if first:
                self._add_hub_edge(graph, first, "cluster_publishing")

        # --- Listening ---
        if include_listening:
            listen_graph = self.listening_viz.create_visualization(
                type(agent.listening), **kwargs
            )
            cluster = self._wrap_in_cluster(
                "listening",
                "Listening",
                listen_graph,
                border_color=self._COLORS["cluster_listen_border"],
                bg_color=self._COLORS["cluster_listen_bg"],
            )
            graph.add_subgraph(cluster)
            first = self._first_node_id(listen_graph)
            if first:
                self._add_hub_edge(graph, first, "cluster_listening")

        # --- Components ---
        if include_components:
            first = self._add_components(agent, graph)
            if first:
                self._add_hub_edge(graph, first, "cluster_components")

        return graph

    def visualize(
        self,
        target: Any,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create and optionally save a comprehensive visualization."""
        graph = self.create_visualization(target, **kwargs)

        if save_file:
            fname = filename or f"{target.__class__.__name__}_comprehensive"
            self.save_diagram(graph, fname, output_dir)

        return graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_hub_edge(
        self,
        graph: pydot.Dot,
        target_node: str,
        cluster_name: str,
    ) -> None:
        """Add a connector edge from the hub node into a section cluster."""
        graph.add_edge(
            pydot.Edge(
                self._HUB_NODE_ID,
                target_node,
                lhead=cluster_name,
                color=Theme.LIGHT_GRAY_BORDER,
                arrowhead="none",
                penwidth="2",
                style="tapered",
            )
        )

    @staticmethod
    def _first_node_id(source_graph: pydot.Dot) -> str | None:
        """Return the ID of the first node in a graph, or ``None``."""
        nodes = source_graph.get_node_list()
        if nodes:
            name: str = nodes[0].get_name().strip('"')
            return name
        return None

    def _wrap_in_cluster(
        self,
        name: str,
        label: str,
        source_graph: pydot.Dot,
        border_color: str,
        bg_color: str,
    ) -> pydot.Cluster:
        """Wrap nodes and edges from a sub-visualization into a cluster."""
        cluster = pydot.Cluster(
            name,
            label=label,
            style="rounded",
            color=border_color,
            bgcolor=bg_color,
            fontname=self.FONT_NAME,
            fontsize=self.FONT_SIZE,
            fontcolor=self._COLORS["cluster_label"],
            penwidth="1.5",
            labeljust="l",
            labelloc="t",
            margin="12",
        )

        for node in source_graph.get_node_list():
            cluster.add_node(node)
        for edge in source_graph.get_edge_list():
            cluster.add_edge(edge)
        for subgraph in source_graph.get_subgraph_list():
            cluster.add_subgraph(subgraph)

        return cluster

    def _add_components(self, agent: Any, graph: pydot.Dot) -> str | None:
        """Discover and add agent sub-components as a cluster.

        Returns the first component node ID, or ``None`` if no components.
        """
        components = agent.get_components() if hasattr(agent, "get_components") else []
        if not components:
            return None

        cluster = pydot.Cluster(
            "components",
            label="Components",
            style="rounded",
            color=self._COLORS["cluster_comp_border"],
            bgcolor=self._COLORS["cluster_comp_bg"],
            fontname=self.FONT_NAME,
            fontsize=self.FONT_SIZE,
            fontcolor=self._COLORS["cluster_label"],
            penwidth="1.5",
            labeljust="l",
            margin="16",
        )

        first_id: str | None = None
        for idx, component in enumerate(components):
            comp_label = (
                component.name
                if hasattr(component, "name")
                else component.__class__.__name__
            )
            comp_node_id = f"comp_{idx}_{comp_label}"
            if first_id is None:
                first_id = comp_node_id
            cluster.add_node(
                self.create_node(
                    node_id=comp_node_id,
                    label=comp_label,
                    fillcolor=self._COLORS["component_fill"],
                    color=self._COLORS["component_border"],
                )
            )

        graph.add_subgraph(cluster)
        return first_id

    # ------------------------------------------------------------------
    # Single-component convenience methods
    # ------------------------------------------------------------------

    def visualize_state_machine_only(
        self,
        agent: "Agent",
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Visualize only the state machine component."""
        return self.state_machine_viz.visualize_with_current_state(
            agent.machine,
            save_file=save_file,
            filename=filename or f"{agent.__class__.__name__}_state_machine",
            output_dir=output_dir,
            **kwargs,
        )

    def visualize_publishing_only(
        self,
        agent: "Agent",
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Visualize only the publishing component."""
        return self.publishing_viz.visualize(
            type(agent.publishing),
            save_file=save_file,
            filename=filename or f"{agent.__class__.__name__}_publishing",
            output_dir=output_dir,
            **kwargs,
        )

    def visualize_listening_only(
        self,
        agent: "Agent",
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Visualize only the listening component."""
        return self.listening_viz.visualize(
            type(agent.listening),
            save_file=save_file,
            filename=filename or f"{agent.__class__.__name__}_listening",
            output_dir=output_dir,
            **kwargs,
        )

    def create_component_diagrams(
        self,
        agent: "Agent",
        save_files: bool = True,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> dict[str, pydot.Dot]:
        """Create separate diagrams for each component."""
        diagrams = {}

        diagrams["state_machine"] = self.visualize_state_machine_only(
            agent, save_file=save_files, output_dir=output_dir, **kwargs
        )

        diagrams["publishing"] = self.visualize_publishing_only(
            agent, save_file=save_files, output_dir=output_dir, **kwargs
        )

        diagrams["listening"] = self.visualize_listening_only(
            agent, save_file=save_files, output_dir=output_dir, **kwargs
        )

        diagrams["comprehensive"] = self.visualize(
            agent,
            save_file=save_files,
            filename=f"{agent.__class__.__name__}_comprehensive",
            output_dir=output_dir,
            **kwargs,
        )

        return diagrams
