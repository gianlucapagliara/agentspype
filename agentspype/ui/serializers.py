"""Serialize agent class-level metadata to JSON-compatible dicts."""

from __future__ import annotations

from typing import Any


def serialize_state_machine(sm_class: type) -> dict[str, Any]:
    """Extract state machine definition from class-level metadata.

    Reads attributes set by ``StateMachineMeta`` at class definition time
    — no instance required.
    """
    states: list[dict[str, Any]] = []
    enter_hooks: dict[str, str] = getattr(sm_class, "_enter_hooks", {})
    exit_hooks: dict[str, str] = getattr(sm_class, "_exit_hooks", {})

    for state in getattr(sm_class, "_states", []):
        states.append(
            {
                "id": state.id,
                "name": state.name or state.id,
                "initial": bool(state.initial),
                "final": bool(state.final),
                "enter_hook": enter_hooks.get(state.id),
                "exit_hook": exit_hooks.get(state.id),
            }
        )

    event_hooks: dict[str, str] = getattr(sm_class, "_event_hooks", {})
    before_event_hooks: dict[str, str] = getattr(sm_class, "_before_event_hooks", {})
    after_event_hooks: dict[str, str] = getattr(sm_class, "_after_event_hooks", {})

    transitions: list[dict[str, Any]] = []
    transition_map: dict[tuple[str, str], list[Any]] = getattr(
        sm_class, "_transition_map", {}
    )
    seen_edges: set[tuple[str, str, str]] = set()

    for (source_id, event_name), trans_list in transition_map.items():
        for t in trans_list:
            target_id = t.target.id if not t.internal else source_id
            edge_key = (source_id, target_id, event_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            transitions.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "event": event_name,
                    "internal": bool(t.internal),
                    "guards": list(t.cond) if t.cond else [],
                    "unless_guards": list(t.unless) if t.unless else [],
                    "hooks": {
                        "before": before_event_hooks.get(event_name),
                        "on": event_hooks.get(event_name),
                        "after": after_event_hooks.get(event_name),
                    },
                }
            )

    all_events = sorted(getattr(sm_class, "_all_event_names", set()))

    return {
        "class_name": sm_class.__name__,
        "states": states,
        "transitions": transitions,
        "events": all_events,
    }


def serialize_publishing(pub_class: type) -> dict[str, Any]:
    """Extract publishing event definitions from class-level metadata."""
    events: list[dict[str, Any]] = []

    if hasattr(pub_class, "get_event_definitions"):
        event_defs = pub_class.get_event_definitions()
        for event_name, event_pub in event_defs.items():
            event_class = getattr(event_pub, "event_class", None)
            event_tag = getattr(event_pub, "event_tag", None)
            events.append(
                {
                    "name": event_name,
                    "tag": _format_tag(event_tag),
                    "event_class": event_class.__name__ if event_class else None,
                }
            )

    return {
        "class_name": pub_class.__name__,
        "events": events,
    }


def serialize_listening(listen_class: type) -> dict[str, Any]:
    """Extract listening subscription definitions from class-level metadata."""
    subscriptions: list[dict[str, Any]] = []

    if hasattr(listen_class, "get_event_definitions"):
        event_defs = listen_class.get_event_definitions()
        for sub_name, details in event_defs.items():
            if isinstance(details, dict):
                publisher_class = details.get("publisher_class")
                event_tag = details.get("event_tag")
            else:
                publisher_class = getattr(details, "publisher_class", None)
                event_tag = getattr(details, "event_tag", None)

            subscriptions.append(
                {
                    "name": sub_name,
                    "event_tag": _format_tag(event_tag),
                    "publisher_class": (
                        publisher_class.__name__
                        if publisher_class and hasattr(publisher_class, "__name__")
                        else str(publisher_class)
                        if publisher_class
                        else None
                    ),
                }
            )

    return {
        "class_name": listen_class.__name__,
        "subscriptions": subscriptions,
    }


def serialize_agent_class(agent_class: type) -> dict[str, Any]:
    """Serialize a full agent class definition to a JSON-compatible dict."""
    definition = agent_class.definition  # type: ignore[attr-defined]

    sm_data = serialize_state_machine(definition.state_machine_class)
    pub_data = serialize_publishing(definition.events_publishing_class)
    listen_data = serialize_listening(definition.events_listening_class)

    config_schema: dict[str, Any] = {}
    try:
        config_schema = definition.configuration_class.model_json_schema()
    except Exception:
        pass

    status_schema: dict[str, Any] = {}
    try:
        status_schema = definition.status_class.model_json_schema()
    except Exception:
        pass

    return {
        "name": agent_class.__name__,
        "module": agent_class.__module__,
        "state_machine": sm_data,
        "publishing": pub_data,
        "listening": listen_data,
        "configuration_schema": config_schema,
        "status_schema": status_schema,
    }


def serialize_cross_agent_relationships(
    agent_classes: list[type],
) -> dict[str, Any]:
    """Build cross-agent event wiring from class-level metadata."""
    pub_to_agent = _build_pub_to_agent_map(agent_classes)
    event_wiring: list[dict[str, str | None]] = []
    seen_edges: set[tuple[str, str, str | None]] = set()

    for cls in agent_classes:
        if not hasattr(cls, "definition"):
            continue
        listen_cls = cls.definition.events_listening_class
        if not hasattr(listen_cls, "get_event_definitions"):
            continue

        listener_name = cls.__name__
        for sub_name, details in listen_cls.get_event_definitions().items():
            _collect_wiring(
                details,
                sub_name,
                listener_name,
                pub_to_agent,
                seen_edges,
                event_wiring,
            )

    return {
        "agents": [cls.__name__ for cls in agent_classes],
        "event_wiring": event_wiring,
    }


def _build_pub_to_agent_map(agent_classes: list[type]) -> dict[type, str]:
    """Map publishing class (and its MRO bases) to owning agent name."""
    pub_to_agent: dict[type, str] = {}
    for cls in agent_classes:
        if not hasattr(cls, "definition"):
            continue
        pub_cls = cls.definition.events_publishing_class
        pub_to_agent[pub_cls] = cls.__name__
        for base in pub_cls.__mro__:
            if base not in pub_to_agent:
                pub_to_agent[base] = cls.__name__
    return pub_to_agent


def _collect_wiring(
    details: Any,
    sub_name: str,
    listener_name: str,
    pub_to_agent: dict[type, str],
    seen_edges: set[tuple[str, str, str | None]],
    event_wiring: list[dict[str, str | None]],
) -> None:
    """Process a single subscription and append to event_wiring if new."""
    if isinstance(details, dict):
        publisher_class = details.get("publisher_class")
        event_tag = details.get("event_tag")
    else:
        publisher_class = getattr(details, "publisher_class", None)
        event_tag = getattr(details, "event_tag", None)

    if publisher_class is None:
        return

    publisher_agent = pub_to_agent.get(publisher_class)
    tag_label = _format_tag(event_tag) if event_tag else sub_name

    edge_key = (publisher_agent or "external", listener_name, tag_label)
    if edge_key in seen_edges:
        return
    seen_edges.add(edge_key)

    event_wiring.append(
        {
            "publisher_agent": publisher_agent,
            "listener_agent": listener_name,
            "event_tag": tag_label,
            "external_publisher": (
                publisher_class.__name__
                if publisher_agent is None and hasattr(publisher_class, "__name__")
                else None
            ),
        }
    )


def _format_tag(tag: Any) -> str | None:
    """Format an event tag for display."""
    if tag is None:
        return None
    if isinstance(tag, list):
        return ", ".join(_format_single_tag(t) for t in tag)
    return _format_single_tag(tag)


def _format_single_tag(tag: Any) -> str:
    """Format a single event tag value."""
    if hasattr(tag, "value"):
        return str(tag.value)
    return str(tag)
