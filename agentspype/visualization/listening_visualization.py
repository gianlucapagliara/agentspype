"""Listening visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization

if TYPE_CHECKING:
    pass


class ListeningVisualization(BaseVisualization):
    """Visualization for agent event listening.

    Publisher deduplication
    ----------------------
    When multiple event subscriptions reference the same publisher class, a
    single publisher node is created in the graph (keyed by the class name).
    However, each subscription still produces its own edge from that publisher
    node to the corresponding callback node.  This keeps the
    diagram compact while faithfully showing every subscription relationship.
    """

    _COLORS = {
        "listener_fill": "#dae8fc",  # soft blue (receiver)
        "listener_border": "#6c8ebf",
        "callback_fill": "#d5e8d4",  # soft green (action handler)
        "callback_border": "#82b366",
        "publisher_fill": "#e1d5e7",  # soft purple (external source)
        "publisher_border": "#9673a6",
        "empty_fill": "#ecf0f1",  # light gray (placeholder)
        "empty_border": "#95a5a6",
        "edge_calls": "#6c8ebf",  # blue for listener -> callback
        "edge_publishes": "#9673a6",  # purple for publisher -> callback
        "edge_label": "#636e72",  # muted gray
    }

    def create_visualization(
        self, target: Any, graph: pydot.Dot | None = None, **kwargs: Any
    ) -> pydot.Dot:
        """Create a visualization of the listening system."""
        listening_class = target if isinstance(target, type) else type(target)

        if hasattr(listening_class, "get_event_definitions"):
            event_definitions = listening_class.get_event_definitions()
        else:
            event_definitions = {}

        if graph is None:
            graph = self.create_graph(f"{listening_class.__name__} Listening")

        # Create the main listener node
        listener_node = self.create_node(
            node_id=listening_class.__name__,
            label=listening_class.__name__,
            fillcolor=self._COLORS["listener_fill"],
            color=self._COLORS["listener_border"],
        )
        graph.add_node(listener_node)

        if not event_definitions:
            self._add_empty_placeholder(graph, listening_class.__name__)
        else:
            self._add_subscriptions(graph, listening_class.__name__, event_definitions)

        return graph

    def _add_empty_placeholder(self, graph: pydot.Dot, listener_name: str) -> None:
        """Add a 'No Subscriptions' placeholder node."""
        no_events_node = self.create_node(
            node_id="no_subscriptions",
            label="No Subscriptions",
            fillcolor=self._COLORS["empty_fill"],
            color=self._COLORS["empty_border"],
            style="filled, dashed",
        )
        graph.add_node(no_events_node)
        graph.add_edge(
            self.create_edge(
                source=listener_name,
                target="no_subscriptions",
                label="listens to",
                style="dashed",
                color=self._COLORS["empty_border"],
            )
        )

    def _add_subscriptions(
        self,
        graph: pydot.Dot,
        listener_name: str,
        event_definitions: dict[str, Any],
    ) -> None:
        """Add subscription nodes and edges for each event definition."""
        seen_publishers: set[str] = set()

        for subscription_name, details in event_definitions.items():
            # Callback node + edge from listener
            graph.add_node(
                self.create_node(
                    node_id=subscription_name,
                    label=subscription_name,
                    fillcolor=self._COLORS["callback_fill"],
                    color=self._COLORS["callback_border"],
                )
            )
            graph.add_edge(
                self.create_edge(
                    source=listener_name,
                    target=subscription_name,
                    label="calls",
                    color=self._COLORS["edge_calls"],
                )
            )

            publisher_class = (
                details.get("publisher_class") if isinstance(details, dict) else None
            )

            self._add_publisher(
                graph,
                publisher_class,
                seen_publishers,
                subscription_name,
            )

    def _add_publisher(
        self,
        graph: pydot.Dot,
        publisher_class: Any,
        seen_publishers: set[str],
        target_node_id: str,
    ) -> None:
        """Add a publisher node (deduplicated) and edge to the target."""
        if not publisher_class:
            return

        publisher_name = (
            publisher_class.__name__
            if hasattr(publisher_class, "__name__")
            else str(publisher_class)
        )
        publisher_node_id = f"pub_{publisher_name}"

        if publisher_name not in seen_publishers:
            seen_publishers.add(publisher_name)
            graph.add_node(
                self.create_node(
                    node_id=publisher_node_id,
                    label=publisher_name,
                    fillcolor=self._COLORS["publisher_fill"],
                    color=self._COLORS["publisher_border"],
                )
            )

        graph.add_edge(
            self.create_edge(
                source=publisher_node_id,
                target=target_node_id,
                label="publishes",
                color=self._COLORS["edge_publishes"],
            )
        )
