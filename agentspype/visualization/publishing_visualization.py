"""Publishing visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization

if TYPE_CHECKING:
    pass


class PublishingVisualization(BaseVisualization):
    """Visualization for agent event publishing."""

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
            fillcolor="lightgreen",
            color="darkgreen",
            final=True,
        )
        graph.add_node(publisher_node)

        # Process event definitions
        if not event_definitions:
            # No events to publish
            no_events_node = self.create_node(
                node_id="no_events",
                label="No Events",
                fillcolor="lightgray",
                color="gray",
                style="rounded, filled, dotted",
            )
            graph.add_node(no_events_node)

            # Add edge showing no publications
            no_events_edge = self.create_edge(
                source=publishing_class.__name__,
                target="no_events",
                label="publishes",
                style="dotted",
                color="gray",
            )
            graph.add_edge(no_events_edge)
        else:
            # Process each event publication
            for event_name, event_publication in event_definitions.items():
                # Create event node
                event_node = self.create_node(
                    node_id=event_name,
                    label=event_name,
                    fillcolor="lightblue",
                    color="darkblue",
                )
                graph.add_node(event_node)

                # Create edge from publisher to event
                publish_edge = self.create_edge(
                    source=publishing_class.__name__,
                    target=event_name,
                    label="publishes",
                    style="solid",
                    color="darkgreen",
                )
                graph.add_edge(publish_edge)

                # Get event tag and class information
                event_tag = getattr(event_publication, "event_tag", None)
                event_class = getattr(event_publication, "event_class", None)

                # Create event tag node if available
                if event_tag:
                    tag_str = str(event_tag)
                    if hasattr(event_tag, "value"):
                        tag_str = str(event_tag.value)

                    tag_node = self.create_node(
                        node_id=f"tag_{tag_str}",
                        label=tag_str,
                        fillcolor="lightyellow",
                        color="orange",
                    )
                    graph.add_node(tag_node)

                    # Create edge from event to tag
                    tag_edge = self.create_edge(
                        source=event_name,
                        target=f"tag_{tag_str}",
                        label="tagged as",
                        style="dashed",
                        color="orange",
                    )
                    graph.add_edge(tag_edge)

                # Create event class node if available
                if event_class:
                    class_name = event_class.__name__
                    class_node = self.create_node(
                        node_id=f"class_{class_name}",
                        label=class_name,
                        fillcolor="lightcyan",
                        color="darkblue",
                    )
                    graph.add_node(class_node)

                    # Create edge from event to class
                    class_edge = self.create_edge(
                        source=event_name,
                        target=f"class_{class_name}",
                        label="data type",
                        style="dotted",
                        color="darkblue",
                    )
                    graph.add_edge(class_edge)

        return graph
