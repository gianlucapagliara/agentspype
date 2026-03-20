"""Listening visualization for agentspype."""

from typing import TYPE_CHECKING, Any

import pydot

from .base_visualization import BaseVisualization

if TYPE_CHECKING:
    pass


class ListeningVisualization(BaseVisualization):
    """Visualization for agent event listening."""

    def create_visualization(
        self, target: Any, graph: pydot.Dot | None = None, **kwargs: Any
    ) -> pydot.Dot:
        """Create a visualization of the listening system."""
        # Get the listening class
        if isinstance(target, type):
            listening_class = target
        else:
            listening_class = type(target)

        # Get event definitions - use hasattr to check method exists
        if hasattr(listening_class, "get_event_definitions"):
            event_definitions = listening_class.get_event_definitions()
        else:
            event_definitions = {}

        # Create or use existing graph
        if graph is None:
            graph = self.create_graph(f"{listening_class.__name__} Listening")

        # Create the main listener node
        listener_node = self.create_node(
            node_id=listening_class.__name__,
            label=listening_class.__name__,
            fillcolor="lightblue",
            color="darkblue",
            final=True,
        )
        graph.add_node(listener_node)

        # Process event definitions
        if not event_definitions:
            # No events to listen to
            no_events_node = self.create_node(
                node_id="no_subscriptions",
                label="No Subscriptions",
                fillcolor="lightgray",
                color="gray",
                style="rounded, filled, dotted",
            )
            graph.add_node(no_events_node)

            # Add edge showing no subscriptions
            no_events_edge = self.create_edge(
                source=listening_class.__name__,
                target="no_subscriptions",
                label="listens to",
                style="dotted",
                color="gray",
            )
            graph.add_edge(no_events_edge)
        else:
            # Process each event subscription
            for subscription_name, subscription_details in event_definitions.items():
                # Create callback node
                callback_node = self.create_node(
                    node_id=subscription_name,
                    label=subscription_name,
                    fillcolor="lightgreen",
                    color="darkgreen",
                )
                graph.add_node(callback_node)

                # Create edge from listener to callback
                listen_edge = self.create_edge(
                    source=listening_class.__name__,
                    target=subscription_name,
                    label="calls",
                    style="solid",
                    color="darkblue",
                )
                graph.add_edge(listen_edge)

                # Get subscription information
                event_tag = (
                    subscription_details.get("event_tag")
                    if isinstance(subscription_details, dict)
                    else None
                )
                publisher_class = (
                    subscription_details.get("publisher_class")
                    if isinstance(subscription_details, dict)
                    else None
                )

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

                    # Create edge from tag to callback
                    tag_edge = self.create_edge(
                        source=f"tag_{tag_str}",
                        target=subscription_name,
                        label="triggers",
                        style="dashed",
                        color="orange",
                    )
                    graph.add_edge(tag_edge)

                # Create publisher node if available
                if publisher_class:
                    publisher_name = (
                        publisher_class.__name__
                        if hasattr(publisher_class, "__name__")
                        else str(publisher_class)
                    )
                    publisher_node = self.create_node(
                        node_id=f"pub_{publisher_name}",
                        label=publisher_name,
                        fillcolor="lightcyan",
                        color="darkgreen",
                    )
                    graph.add_node(publisher_node)

                    # Create edge from publisher to tag (if tag exists) or callback
                    if event_tag:
                        pub_edge = self.create_edge(
                            source=f"pub_{publisher_name}",
                            target=f"tag_{tag_str}",
                            label="publishes",
                            style="solid",
                            color="darkgreen",
                        )
                    else:
                        pub_edge = self.create_edge(
                            source=f"pub_{publisher_name}",
                            target=subscription_name,
                            label="publishes to",
                            style="solid",
                            color="darkgreen",
                        )
                    graph.add_edge(pub_edge)

        return graph
