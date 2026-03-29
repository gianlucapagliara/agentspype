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
        "start": {"color": "#82b366", "style": "dashed"},  # muted green
        "stop": {"color": "#b85450", "style": "dashed"},  # muted red
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
        "internal_edge": "#7f8c8d",  # muted gray for internal transitions
    }

    def create_visualization(  # noqa: C901
        self,
        target: Any,
        graph: pydot.Dot | None = None,
        current_state: State | None = None,
        edge_style_map: dict[str, dict[str, str]] | None = None,
        show_guards: bool = True,
        show_hooks: bool = True,
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
            show_guards: Whether to show guard conditions (cond/unless) on
                transition edges. Defaults to True.
            show_hooks: Whether to show hook methods (on_enter_*, on_exit_*,
                before_*, on_*, after_*) on states and transitions.
                Defaults to False.
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

        # Collect hook tables for annotation
        enter_hooks = getattr(state_machine_class, "_enter_hooks", {})
        exit_hooks = getattr(state_machine_class, "_exit_hooks", {})

        # Add state nodes
        for state_id, state in states_map.items():
            if state.initial:
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

            # Build label: plain name or record with hooks
            state_label = state.name or state_id
            if show_hooks:
                hooks: list[str] = []
                if state_id in enter_hooks:
                    hooks.append(enter_hooks[state_id])
                if state_id in exit_hooks:
                    hooks.append(exit_hooks[state_id])
                if hooks:
                    hooks_str = "\\l".join(hooks) + "\\l"
                    state_label = f"{{{state_label} | {hooks_str}}}"

            node = pydot.Node(
                state_id,
                label=state_label,
                shape=shape,
                fillcolor=fillcolor,
                style="filled",
                color=color,
                peripheries=peripheries,
                fontname=self.FONT_NAME,
                fontsize=self.FONT_SIZE,
            )
            dot_graph.add_node(node)

        # Collect event hook tables for edge annotation
        event_hooks = getattr(state_machine_class, "_event_hooks", {})
        before_event_hooks = getattr(state_machine_class, "_before_event_hooks", {})
        after_event_hooks = getattr(state_machine_class, "_after_event_hooks", {})

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

                # Internal transitions: dotted + muted color
                if t.internal:
                    edge_attrs["style"] = "dotted"
                    edge_attrs["color"] = self._COLORS["internal_edge"]

                # Apply edge styles (explicit styles override internal defaults)
                if event_name in effective_edge_styles:
                    style_attrs = effective_edge_styles[event_name]
                    if "color" in style_attrs:
                        edge_attrs["color"] = style_attrs["color"]
                    if "style" in style_attrs:
                        edge_attrs["style"] = style_attrs["style"]
                    if "penwidth" in style_attrs:
                        edge_attrs["penwidth"] = style_attrs["penwidth"]
                elif (
                    not t.internal
                    and event_name in state_machine_class._all_event_names
                ):
                    edge_attrs["color"] = self._COLORS["edge_event"]
                    edge_attrs["style"] = "solid"

                # Build edge label
                label = event_name

                # Append guard conditions
                if show_guards:
                    guard_parts: list[str] = []
                    if t.cond:
                        guard_parts.extend(t.cond)
                    if t.unless:
                        guard_parts.extend(f"!{u}" for u in t.unless)
                    if guard_parts:
                        label += f" [{', '.join(guard_parts)}]"

                # Append hook annotations
                if show_hooks:
                    hook_names: list[str] = []
                    if event_name in before_event_hooks:
                        hook_names.append(before_event_hooks[event_name])
                    if event_name in event_hooks:
                        hook_names.append(event_hooks[event_name])
                    if event_name in after_event_hooks:
                        hook_names.append(after_event_hooks[event_name])
                    if hook_names:
                        label += "\\n" + "\\n".join(hook_names)

                edge = pydot.Edge(
                    source_id,
                    target_id,
                    label=label,
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
