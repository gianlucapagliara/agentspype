"""Base visualization functionality for agentspype."""

import os
from abc import ABC, abstractmethod
from typing import Any

import pydot


class BaseVisualization(ABC):
    """Base class for all visualization components."""

    # Style constants
    FONT_NAME = "Arial"
    FONT_SIZE = "10pt"
    GRAPH_RANKDIR = "LR"

    @classmethod
    def create_graph(cls, name: str, graph_type: str = "digraph") -> pydot.Dot:
        """Create a new pydot graph with default styling."""
        graph = pydot.Dot(
            name,
            graph_type=graph_type,
            fontname=cls.FONT_NAME,
            fontsize=cls.FONT_SIZE,
            rankdir=cls.GRAPH_RANKDIR,
            bgcolor="transparent",
            pad="0.5",
        )
        return graph

    @classmethod
    def create_node(
        cls,
        node_id: str,
        label: str,
        shape: str = "Mrecord",
        style: str = "filled",
        peripheries: int = 1,
        fillcolor: str = "white",
        color: str = "black",
    ) -> pydot.Node:
        """Create a styled node for the graph.

        Args:
            node_id: Unique identifier for the node within the graph.
            label: Display text shown inside the node.
            shape: Graphviz node shape (e.g. ``"Mrecord"``, ``"rectangle"``).
            style: Comma-separated Graphviz style attributes
                (e.g. ``"filled"``).
            peripheries: Number of node borders (``2`` for final/terminal).
            fillcolor: Background fill colour of the node.
            color: Border colour of the node.
        """
        node = pydot.Node(
            node_id,
            label=label,
            shape=shape,
            style=style,
            fontname=cls.FONT_NAME,
            fontsize=cls.FONT_SIZE,
            peripheries=peripheries,
            fillcolor=fillcolor,
            color=color,
        )
        return node

    @classmethod
    def create_edge(
        cls,
        source: str,
        target: str,
        label: str = "",
        style: str = "solid",
        color: str = "#2d3436",
        arrowhead: str = "vee",
        penwidth: str = "1.2",
        fontcolor: str = "#636e72",
    ) -> pydot.Edge:
        """Create a styled edge for the graph."""
        edge = pydot.Edge(
            source,
            target,
            label=label,
            color=color,
            style=style,
            fontname=cls.FONT_NAME,
            fontsize=cls.FONT_SIZE,
            arrowhead=arrowhead,
            penwidth=penwidth,
            fontcolor=fontcolor,
        )
        return edge

    @classmethod
    def save_diagram(
        cls, graph: pydot.Dot, filename: str, output_dir: str = ".diagrams"
    ) -> str:
        """Save the diagram to a file."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create full path
        if not filename.endswith(".png"):
            filename += ".png"
        full_path = os.path.join(output_dir, filename)

        # Save the diagram using getattr to handle missing type stubs
        graph.write_png(full_path)
        print(f"Diagram saved to: {full_path}")
        return full_path

    @abstractmethod
    def create_visualization(
        self, target: Any, graph: pydot.Dot | None = None, **kwargs: Any
    ) -> pydot.Dot:
        """Create a visualization for the target object."""
        pass

    def visualize(
        self,
        target: Any,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create and optionally save a visualization."""
        graph = self.create_visualization(target, **kwargs)

        if save_file:
            if filename is None:
                filename = f"{target.__class__.__name__}_visualization"
            self.save_diagram(graph, filename, output_dir)

        return graph
