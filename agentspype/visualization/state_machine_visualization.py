"""State machine visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from agentspype.fsm import State

from .base_visualization import BaseVisualization

if TYPE_CHECKING:
    from agentspype.agent.state_machine import AgentStateMachine


class StateMachineVisualization(BaseVisualization):
    """Visualization for agent state machines."""

    # Default edge style map for common transitions
    DEFAULT_EDGE_STYLE_MAP: dict[str, dict[str, str]] = {
        "start": {"color": "green", "style": "dashed"},
        "stop": {"color": "red", "style": "dashed"},
    }

    # Color palette
    _COLORS = {
        "initial_fill": "#d5e8d4",  # soft green
        "initial_border": "#82b366",
        "normal_fill": "#dae8fc",  # soft blue
        "normal_border": "#6c8ebf",
        "final_fill": "#f8cecc",  # soft red
        "final_border": "#b85450",
        "highlight_fill": "#fff2cc",  # soft gold
        "highlight_border": "#d6b656",
        "edge_default": "#2d3436",
        "edge_event": "#2d3436",
    }

    def create_visualization(  # noqa: C901
        self,
        target: Any,
        graph: pydot.Dot | None = None,
        current_state: State | None = None,
        edge_style_map: dict[str, dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> pydot.Dot:
        """Create a visualization of the state machine.

        Args:
            target: The state machine instance to visualize.
            graph: An existing graph to add to, or None to create a new one.
            current_state: The current state to highlight, or None.
            edge_style_map: A mapping of transition event names to style attributes.
                Each value is a dict with optional keys: "color", "style", "penwidth".
                Default map styles "start" as green/dashed and "stop" as red/dashed.
                Pass a custom dict to override or extend the defaults.
            **kwargs: Additional arguments.

        Returns:
            pydot.Dot: The generated state machine diagram.
        """
        state_machine = target
        state_machine_class = state_machine.__class__

        # Merge default edge styles with user-provided ones
        effective_edge_styles = dict(self.DEFAULT_EDGE_STYLE_MAP)
        if edge_style_map is not None:
            effective_edge_styles.update(edge_style_map)

        # Build dot graph from our FSM data
        dot_graph = pydot.Dot(graph_type="digraph")
        dot_graph.obj_dict["attributes"]["rankdir"] = self.GRAPH_RANKDIR
        dot_graph.obj_dict["attributes"]["fontname"] = self.FONT_NAME
        dot_graph.obj_dict["attributes"]["fontsize"] = self.FONT_SIZE
        dot_graph.obj_dict["attributes"]["bgcolor"] = "transparent"
        dot_graph.obj_dict["attributes"]["pad"] = "0.5"

        states_map = state_machine_class.states_map

        # Add initial marker node (standard FSM convention: small black dot)
        initial_node = pydot.Node(
            "i",
            shape="point",
            width="0.2",
            height="0.2",
            color=self._COLORS["initial_border"],
            fillcolor=self._COLORS["initial_border"],
        )
        dot_graph.add_node(initial_node)

        # Add state nodes
        initial_state = None
        for state_id, state in states_map.items():
            if state.initial:
                initial_state = state
                fillcolor = self._COLORS["initial_fill"]
                color = self._COLORS["initial_border"]
                shape = "Mrecord"
                peripheries = "1"
            elif state.final:
                fillcolor = self._COLORS["final_fill"]
                color = self._COLORS["final_border"]
                shape = "Mrecord"
                peripheries = "2"  # double border = final state convention
            else:
                fillcolor = self._COLORS["normal_fill"]
                color = self._COLORS["normal_border"]
                shape = "Mrecord"
                peripheries = "1"

            # Highlight current state if provided
            if current_state and state_id == current_state.id:
                fillcolor = self._COLORS["highlight_fill"]
                color = self._COLORS["highlight_border"]
                peripheries = "2"

            node = pydot.Node(
                state_id,
                label=state.name or state_id,
                shape=shape,
                fillcolor=fillcolor,
                style="filled",
                color=color,
                peripheries=peripheries,
                fontname=self.FONT_NAME,
                fontsize=self.FONT_SIZE,
            )
            dot_graph.add_node(node)

        # Add edge from initial marker to initial state
        if initial_state:
            dot_graph.add_edge(
                pydot.Edge(
                    "i",
                    initial_state.id,
                    arrowhead="vee",
                    color=self._COLORS["initial_border"],
                    penwidth="1.5",
                )
            )

        # Add transition edges
        transition_map = state_machine_class._transition_map
        seen_edges: set[tuple[str, str, str]] = set()
        for (source_id, event_name), transitions in transition_map.items():
            for t in transitions:
                target_id = t.target.id if not t.internal else source_id
                edge_key = (source_id, target_id, event_name)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                edge_attrs: dict[str, str] = {
                    "fontname": self.FONT_NAME,
                    "fontsize": str(self.FONT_SIZE),
                    "color": self._COLORS["edge_default"],
                    "fontcolor": "#636e72",
                    "arrowhead": "vee",
                    "penwidth": "1.2",
                }

                # Apply edge styles
                if event_name in effective_edge_styles:
                    style_attrs = effective_edge_styles[event_name]
                    if "color" in style_attrs:
                        edge_attrs["color"] = style_attrs["color"]
                    if "style" in style_attrs:
                        edge_attrs["style"] = style_attrs["style"]
                    if "penwidth" in style_attrs:
                        edge_attrs["penwidth"] = style_attrs["penwidth"]
                elif event_name in state_machine_class._all_event_names:
                    edge_attrs["color"] = self._COLORS["edge_event"]
                    edge_attrs["style"] = "solid"

                edge = pydot.Edge(
                    source_id,
                    target_id,
                    label=event_name,
                    **edge_attrs,
                )
                dot_graph.add_edge(edge)

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
