"""Publishing visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization

if TYPE_CHECKING:
    pass


class PublishingVisualization(BaseVisualization):
    """Visualization for agent event publishing."""

    _COLORS = {
        "publisher_fill": "#d5e8d4",  # soft green (source)
        "publisher_border": "#82b366",
        "event_fill": "#dae8fc",  # soft blue (data)
        "event_border": "#6c8ebf",
        "empty_fill": "#ecf0f1",  # light gray (placeholder)
        "empty_border": "#95a5a6",
        "edge_publish": "#82b366",  # green for publish relationship
        "edge_label": "#636e72",  # muted gray
    }

    def create_visualization(
        self, target: Any, graph: pydot.Dot | None = None, **kwargs: Any
    ) -> pydot.Dot:
        """Create a visualization of the publishing system."""
        # Get the publishing class
        if isinstance(target, type):
            publishing_class = target
        else:
            publishing_class = type(target)

        # Get event definitions - use hasattr to check method exists
        if hasattr(publishing_class, "get_event_definitions"):
            event_definitions = publishing_class.get_event_definitions()
        else:
            event_definitions = {}

        # Create or use existing graph
        if graph is None:
            graph = self.create_graph(f"{publishing_class.__name__} Publishing")

        # Create the main publisher node
        publisher_node = self.create_node(
            node_id=publishing_class.__name__,
            label=publishing_class.__name__,
            fillcolor=self._COLORS["publisher_fill"],
            color=self._COLORS["publisher_border"],
        )
        graph.add_node(publisher_node)

        # Process event definitions
        if not event_definitions:
            # No events to publish
            no_events_node = self.create_node(
                node_id="no_events",
                label="No Events",
                fillcolor=self._COLORS["empty_fill"],
                color=self._COLORS["empty_border"],
                style="filled, dashed",
            )
            graph.add_node(no_events_node)

            # Add edge showing no publications
            no_events_edge = self.create_edge(
                source=publishing_class.__name__,
                target="no_events",
                label="publishes",
                style="dashed",
                color=self._COLORS["empty_border"],
            )
            graph.add_edge(no_events_edge)
        else:
            # Process each event publication
            for event_name, event_publication in event_definitions.items():
                # Use enum member name when available, fallback to attribute name
                original_tag = getattr(event_publication, "original_tag", None)
                display_name = (
                    original_tag.name if hasattr(original_tag, "name") else event_name
                )

                # Build record label: event name + data type if available
                event_class = getattr(event_publication, "event_class", None)
                if event_class:
                    label = f"{{{display_name} | {event_class.__name__}}}"
                else:
                    label = display_name

                event_node = self.create_node(
                    node_id=event_name,
                    label=label,
                    fillcolor=self._COLORS["event_fill"],
                    color=self._COLORS["event_border"],
                )
                graph.add_node(event_node)

                # Create edge from publisher to event
                publish_edge = self.create_edge(
                    source=publishing_class.__name__,
                    target=event_name,
                    label="publishes",
                    color=self._COLORS["edge_publish"],
                )
                graph.add_edge(publish_edge)

        return graph
