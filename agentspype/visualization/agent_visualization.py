"""Comprehensive agent visualization for agentspype."""

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
        show_current_state: bool = True,
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create a comprehensive visualization of the agent."""
        # Type check the target
        agent = target

        # Create the main graph
        if graph is None:
            graph = self.create_graph(f"{agent.__class__.__name__} Agent")

        # Clear the graph label for cleaner look
        graph.obj_dict["attributes"]["label"] = ""

        # Create the main agent node
        agent_node = self.create_node(
            node_id=agent.__class__.__name__,
            label=agent.__class__.__name__,
            fillcolor="lightpink",
            color="darkred",
            final=True,
        )
        graph.add_node(agent_node)

        # Add state machine visualization if requested
        if include_state_machine:
            current_state = agent.machine.current_state if show_current_state else None
            state_machine_graph = self.state_machine_viz.create_visualization(
                agent.machine, current_state=current_state, **kwargs
            )

            # Merge the state machine graph
            self._merge_graph(graph, state_machine_graph)

            # Connect agent to state machine
            sm_edge = self.create_edge(
                source=agent.__class__.__name__,
                target="i",  # Initial state node from state machine
                label="state machine",
                style="solid",
                color="darkred",
            )
            graph.add_edge(sm_edge)

        # Add publishing visualization if requested
        if include_publishing:
            publishing_graph = self.publishing_viz.create_visualization(
                type(agent.publishing), **kwargs
            )

            # Merge the publishing graph
            self._merge_graph(graph, publishing_graph)

            # Connect agent to publishing
            pub_edge = self.create_edge(
                source=agent.__class__.__name__,
                target=type(agent.publishing).__name__,
                label="publishes via",
                style="solid",
                color="darkgreen",
            )
            graph.add_edge(pub_edge)

        # Add listening visualization if requested
        if include_listening:
            listening_graph = self.listening_viz.create_visualization(
                type(agent.listening), **kwargs
            )

            # Merge the listening graph
            self._merge_graph(graph, listening_graph)

            # Connect agent to listening
            listen_edge = self.create_edge(
                source=agent.__class__.__name__,
                target=type(agent.listening).__name__,
                label="listens via",
                style="solid",
                color="darkblue",
            )
            graph.add_edge(listen_edge)

        return graph

    def _merge_graph(self, target_graph: pydot.Dot, source_graph: pydot.Dot) -> None:
        """Merge nodes and edges from source graph into target graph."""
        # Add all nodes from source graph
        for node in source_graph.get_node_list():
            target_graph.add_node(node)

        # Add all edges from source graph
        for edge in source_graph.get_edge_list():
            target_graph.add_edge(edge)

        # Add all subgraphs from source graph
        for subgraph in source_graph.get_subgraph_list():
            target_graph.add_subgraph(subgraph)

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

        # State machine diagram
        diagrams["state_machine"] = self.visualize_state_machine_only(
            agent, save_file=save_files, output_dir=output_dir, **kwargs
        )

        # Publishing diagram
        diagrams["publishing"] = self.visualize_publishing_only(
            agent, save_file=save_files, output_dir=output_dir, **kwargs
        )

        # Listening diagram
        diagrams["listening"] = self.visualize_listening_only(
            agent, save_file=save_files, output_dir=output_dir, **kwargs
        )

        # Comprehensive diagram
        diagrams["comprehensive"] = self.visualize(
            agent,
            save_file=save_files,
            filename=f"{agent.__class__.__name__}_comprehensive",
            output_dir=output_dir,
            **kwargs,
        )

        return diagrams
