"""State machine visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot
from statemachine import State
from statemachine.contrib.diagram import DotGraphMachine

from .base_visualization import BaseVisualization

if TYPE_CHECKING:
    from agentspype.agent.state_machine import AgentStateMachine


class StateMachineVisualization(BaseVisualization):
    """Visualization for agent state machines."""

    def create_visualization(  # noqa: C901
        self,
        target: Any,
        graph: pydot.Dot | None = None,
        current_state: State | None = None,
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create a visualization of the state machine."""
        # Type check the target
        state_machine = target

        # Get the state machine class for visualization
        state_machine_class = state_machine.__class__

        # Use statemachine's built-in diagram generation
        diagram_generator = DotGraphMachine(state_machine_class)  # type: ignore[no-untyped-call]
        dot_graph = diagram_generator()

        # Apply our custom styling
        dot_graph.obj_dict["attributes"]["rankdir"] = self.GRAPH_RANKDIR
        dot_graph.obj_dict["attributes"]["fontname"] = self.FONT_NAME
        dot_graph.obj_dict["attributes"]["fontsize"] = self.FONT_SIZE

        # Style the nodes
        for node in dot_graph.get_node_list():
            node_name = node.get_name().strip('"')

            # Set basic styling
            node.set_fontname(self.FONT_NAME)
            node.set_fontsize(self.FONT_SIZE)

            # Color nodes based on their type
            if node_name in state_machine_class.states_map:
                # This is a state node
                node.set_fillcolor("lightblue")
                node.set_style("filled,rounded")
                node.set_color("darkblue")
            elif node_name == "i":
                # Initial state marker
                node.set_fillcolor("lightgreen")
                node.set_style("filled")
                node.set_color("darkgreen")
            elif node_name == "end":
                # Final state marker
                node.set_fillcolor("lightcoral")
                node.set_style("filled")
                node.set_color("darkred")
            else:
                # Other nodes
                node.set_fillcolor("lightgray")
                node.set_style("filled")
                node.set_color("gray")

            # Highlight current state if provided
            if current_state and node_name == current_state.id:
                node.set_fillcolor("gold")
                node.set_color("orange")
                node.set_style("filled,rounded,bold")

        # Style the edges
        for edge in dot_graph.get_edge_list():
            edge.set_fontname(self.FONT_NAME)
            edge.set_fontsize(self.FONT_SIZE)
            edge.set_color("darkslategray")

            # Get edge label
            edge_label = edge.get_label().strip('"')

            # Color edges based on their type
            if edge_label == "start":
                edge.set_color("darkgreen")
                edge.set_style("dashed")
            elif edge_label == "stop":
                edge.set_color("darkred")
                edge.set_style("dashed")
            elif edge_label == "":
                # Empty transition
                edge.set_style("dotted")
                edge.set_color("gray")
            elif edge_label in getattr(state_machine_class, "_events", {}):
                # This is a registered event
                edge.set_color("darkblue")
                edge.set_style("bold")

        return dot_graph

    def visualize_with_current_state(
        self,
        state_machine: "AgentStateMachine",
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> pydot.Dot:
        """Visualize the state machine with current state highlighted."""
        current_state = state_machine.current_state
        return self.visualize(
            state_machine,
            save_file=save_file,
            filename=filename,
            output_dir=output_dir,
            current_state=current_state,
            **kwargs,
        )
