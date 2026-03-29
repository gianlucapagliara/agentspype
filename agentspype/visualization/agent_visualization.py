"""Comprehensive agent visualization for agentspype."""

import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization
from .listening_visualization import ListeningVisualization
from .publishing_visualization import PublishingVisualization
from .state_machine_visualization import StateMachineVisualization

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent


class AgentVisualization(BaseVisualization):
    """Comprehensive visualization for agents including state machine, publishing, and listening."""

    _COLORS = {
        "component_fill": "#fff2cc",  # soft gold
        "component_border": "#d6b656",
        "cluster_sm_border": "#b85450",  # soft red
        "cluster_sm_bg": "#f9f9f9",
        "cluster_pub_border": "#82b366",  # green
        "cluster_pub_bg": "#f9f9f9",
        "cluster_listen_border": "#6c8ebf",  # blue
        "cluster_listen_bg": "#f9f9f9",
        "cluster_comp_border": "#d6b656",  # gold
        "cluster_comp_bg": "#f9f9f9",
        "cluster_label": "#636e72",  # muted gray
    }

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
        include_components: bool = True,
        show_current_state: bool = True,
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create a comprehensive visualization of the agent.

        Each included section (state machine, publishing, listening,
        components) is rendered as a labelled cluster subgraph.

        Note: for saved output, ``visualize()`` composites each section
        as an independent LR image stacked vertically, so every section
        keeps its natural horizontal flow while being left-aligned.

        Args:
            target: The agent instance to visualize.
            graph: An existing graph to add to, or None to create a new one.
            include_state_machine: Whether to include state machine visualization.
            include_publishing: Whether to include publishing visualization.
            include_listening: Whether to include listening visualization.
            include_components: Whether to include agent sub-components.
            show_current_state: Whether to highlight the current state.
            **kwargs: Additional arguments passed to sub-visualizers
                (e.g. edge_style_map for state machine).

        Returns:
            pydot.Dot: The generated agent diagram.
        """
        agent = target

        if graph is None:
            graph = self.create_graph(f"{agent.__class__.__name__}_agent")

        graph.obj_dict["attributes"]["rankdir"] = "LR"

        if include_state_machine:
            current_state = agent.machine.current_state if show_current_state else None
            sm_graph = self.state_machine_viz.create_visualization(
                agent.machine, current_state=current_state, **kwargs
            )
            graph.add_subgraph(
                self._wrap_in_cluster(
                    "state_machine",
                    "State Machine",
                    sm_graph,
                    border_color=self._COLORS["cluster_sm_border"],
                    bg_color=self._COLORS["cluster_sm_bg"],
                )
            )

        if include_publishing:
            pub_graph = self.publishing_viz.create_visualization(
                type(agent.publishing), **kwargs
            )
            graph.add_subgraph(
                self._wrap_in_cluster(
                    "publishing",
                    "Publishing",
                    pub_graph,
                    border_color=self._COLORS["cluster_pub_border"],
                    bg_color=self._COLORS["cluster_pub_bg"],
                )
            )

        if include_listening:
            listen_graph = self.listening_viz.create_visualization(
                type(agent.listening), **kwargs
            )
            graph.add_subgraph(
                self._wrap_in_cluster(
                    "listening",
                    "Listening",
                    listen_graph,
                    border_color=self._COLORS["cluster_listen_border"],
                    bg_color=self._COLORS["cluster_listen_bg"],
                )
            )

        if include_components:
            self._add_components(agent, graph)

        return graph

    def visualize(
        self,
        target: Any,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create and optionally save a comprehensive visualization.

        When *save_file* is ``True``, each section is rendered
        independently with its own LR layout, then the images are
        stacked vertically in a master TB graph.  This guarantees
        left-aligned sections with correct horizontal flow inside each.
        """
        graph = self.create_visualization(target, **kwargs)

        if save_file:
            fname = filename or f"{target.__class__.__name__}_comprehensive"
            self._save_composite(target, fname, output_dir, **kwargs)

        return graph

    # ------------------------------------------------------------------
    # Composite rendering
    # ------------------------------------------------------------------

    def _save_composite(
        self,
        agent: Any,
        filename: str,
        output_dir: str,
        **kwargs: Any,
    ) -> str:
        """Render each section as a standalone LR image, then stack them."""
        include_sm = kwargs.pop("include_state_machine", True)
        include_pub = kwargs.pop("include_publishing", True)
        include_listen = kwargs.pop("include_listening", True)
        include_comp = kwargs.pop("include_components", True)
        show_current = kwargs.pop("show_current_state", True)

        tmp_dir = tempfile.mkdtemp(prefix="agentspype_viz_")
        section_paths: list[str] = []

        try:
            if include_sm:
                current = agent.machine.current_state if show_current else None
                sm_graph = self.state_machine_viz.create_visualization(
                    agent.machine, current_state=current, **kwargs
                )
                path = os.path.join(tmp_dir, "sm.png")
                self._render_section(
                    "State Machine",
                    sm_graph,
                    self._COLORS["cluster_sm_border"],
                    self._COLORS["cluster_sm_bg"],
                    path,
                )
                section_paths.append(path)

            if include_pub:
                pub_graph = self.publishing_viz.create_visualization(
                    type(agent.publishing), **kwargs
                )
                path = os.path.join(tmp_dir, "pub.png")
                self._render_section(
                    "Publishing",
                    pub_graph,
                    self._COLORS["cluster_pub_border"],
                    self._COLORS["cluster_pub_bg"],
                    path,
                )
                section_paths.append(path)

            if include_listen:
                listen_graph = self.listening_viz.create_visualization(
                    type(agent.listening), **kwargs
                )
                path = os.path.join(tmp_dir, "listen.png")
                self._render_section(
                    "Listening",
                    listen_graph,
                    self._COLORS["cluster_listen_border"],
                    self._COLORS["cluster_listen_bg"],
                    path,
                )
                section_paths.append(path)

            if include_comp:
                comp_path = self._render_components_section(agent, tmp_dir)
                if comp_path:
                    section_paths.append(comp_path)

            # Assemble sections inside a single agent container
            master = pydot.Dot(
                graph_type="digraph",
                rankdir="TB",
                bgcolor="transparent",
                pad="0.2",
                margin="0",
                ranksep="0.1",
                nodesep="0.1",
            )

            agent_cluster = pydot.Cluster(
                "agent",
                label=agent.__class__.__name__,
                style="rounded,bold",
                color="#2d3436",
                bgcolor="#ffffff",
                fontname=self.FONT_NAME,
                fontsize="14",
                fontcolor="#2d3436",
                penwidth="2",
                labeljust="l",
                labelloc="t",
                margin="16",
            )

            prev_id: str | None = None
            for i, img_path in enumerate(section_paths):
                node_id = f"_sec_{i}"
                agent_cluster.add_node(
                    pydot.Node(
                        node_id,
                        shape="none",
                        image=img_path,
                        label="",
                    )
                )
                if prev_id is not None:
                    agent_cluster.add_edge(pydot.Edge(prev_id, node_id, style="invis"))
                prev_id = node_id

            master.add_subgraph(agent_cluster)
            return self.save_diagram(master, filename, output_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _render_section(
        self,
        label: str,
        source_graph: pydot.Dot,
        border_color: str,
        bg_color: str,
        output_path: str,
    ) -> None:
        """Render a single section with a cluster border to PNG."""
        wrapper = pydot.Dot(
            graph_type="digraph",
            rankdir="LR",
            bgcolor="transparent",
            pad="0.1",
            margin="0",
        )
        wrapper.obj_dict["attributes"]["fontname"] = self.FONT_NAME
        wrapper.obj_dict["attributes"]["fontsize"] = self.FONT_SIZE

        cluster = self._wrap_in_cluster(
            "section", label, source_graph, border_color, bg_color
        )
        wrapper.add_subgraph(cluster)
        wrapper.write_png(output_path)

    def _render_components_section(self, agent: Any, tmp_dir: str) -> str | None:
        """Render components cluster to PNG if the agent has any."""
        components = agent.get_components() if hasattr(agent, "get_components") else []
        if not components:
            return None

        wrapper = pydot.Dot(
            graph_type="digraph",
            rankdir="LR",
            bgcolor="transparent",
            pad="0.1",
            margin="0",
        )
        wrapper.obj_dict["attributes"]["fontname"] = self.FONT_NAME
        wrapper.obj_dict["attributes"]["fontsize"] = self.FONT_SIZE

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
            labelloc="t",
            margin="12",
        )

        for idx, component in enumerate(components):
            comp_label = (
                component.name
                if hasattr(component, "name")
                else component.__class__.__name__
            )
            comp_node_id = f"comp_{idx}_{comp_label}"
            cluster.add_node(
                self.create_node(
                    node_id=comp_node_id,
                    label=comp_label,
                    fillcolor=self._COLORS["component_fill"],
                    color=self._COLORS["component_border"],
                )
            )

        wrapper.add_subgraph(cluster)
        path = os.path.join(tmp_dir, "comp.png")
        wrapper.write_png(path)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _add_components(self, agent: Any, graph: pydot.Dot) -> None:
        """Discover and add agent sub-components as a cluster."""
        components = agent.get_components() if hasattr(agent, "get_components") else []
        if not components:
            return

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

        for idx, component in enumerate(components):
            comp_label = (
                component.name
                if hasattr(component, "name")
                else component.__class__.__name__
            )
            comp_node_id = f"comp_{idx}_{comp_label}"
            cluster.add_node(
                self.create_node(
                    node_id=comp_node_id,
                    label=comp_label,
                    fillcolor=self._COLORS["component_fill"],
                    color=self._COLORS["component_border"],
                )
            )

        graph.add_subgraph(cluster)

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
