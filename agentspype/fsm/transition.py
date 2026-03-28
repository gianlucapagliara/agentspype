"""Transition and TransitionList for the custom FSM."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentspype.fsm.state import State


def _normalize_guard(
    value: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Normalize a guard specification to a list of method names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


class Transition:
    """Single source -> target transition with optional guards."""

    __slots__ = ("source", "target", "event", "internal", "cond", "unless")

    def __init__(
        self,
        source: State,
        target: State,
        *,
        event: str = "",
        internal: bool = False,
        cond: str | list[str] | tuple[str, ...] | None = None,
        unless: str | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.event = event
        self.internal = internal
        self.cond = _normalize_guard(cond)
        self.unless = _normalize_guard(unless)

    def __repr__(self) -> str:
        return (
            f"Transition({self.source.id!r} -> {self.target.id!r}, "
            f"event={self.event!r}, internal={self.internal})"
        )


class TransitionList:
    """Collection of transitions for a single event. Supports ``|`` operator."""

    def __init__(self, transitions: list[Transition] | None = None) -> None:
        self.transitions: list[Transition] = transitions or []

    def add_transitions(self, other: TransitionList) -> None:
        self.transitions.extend(other.transitions)

    def __or__(self, other: TransitionList) -> TransitionList:
        return TransitionList(self.transitions + other.transitions)

    def __repr__(self) -> str:
        return f"TransitionList({self.transitions!r})"
