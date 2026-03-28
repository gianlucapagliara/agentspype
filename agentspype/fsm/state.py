"""State definition with transition-building API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentspype.fsm.transition import TransitionList


class StateTransitions:
    """Lightweight wrapper providing outgoing transition info for a state.

    Populated by the metaclass after the full transition map is built.
    """

    __slots__ = ("_transitions", "unique_events")

    def __init__(
        self,
        transitions: list[Any] | None = None,
        events: frozenset[str] | None = None,
    ) -> None:
        self._transitions = transitions or []
        self.unique_events = events or frozenset()

    @property
    def transitions(self) -> list[Any]:
        return self._transitions

    def add_transitions(self, tl: TransitionList) -> None:
        self._transitions.extend(tl.transitions)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)


class _TransitionBuilder:
    """Fluent API: ``state.to(target, cond=...)`` returns a TransitionList."""

    __slots__ = ("_source",)

    def __init__(self, source: State) -> None:
        self._source = source

    def __call__(
        self,
        target: State,
        *,
        cond: str | list[str] | tuple[str, ...] | None = None,
        unless: str | list[str] | tuple[str, ...] | None = None,
        internal: bool = False,
    ) -> TransitionList:
        from agentspype.fsm.transition import Transition, TransitionList

        t = Transition(
            source=self._source,
            target=target,
            internal=internal,
            cond=cond,
            unless=unless,
        )
        tl = TransitionList([t])
        self._source.transitions.add_transitions(tl)
        return tl

    def itself(
        self,
        *,
        cond: str | list[str] | tuple[str, ...] | None = None,
        unless: str | list[str] | tuple[str, ...] | None = None,
        internal: bool = False,
    ) -> TransitionList:
        return self(self._source, cond=cond, unless=unless, internal=internal)


class State:
    """Immutable state definition with transition-building API.

    Mirrors the ``python-statemachine`` ``State`` class API.
    """

    def __init__(
        self,
        name: str = "",
        *,
        value: str | None = None,
        initial: bool = False,
        final: bool = False,
    ) -> None:
        self.name = name
        self.id: str = ""  # Set by metaclass from attribute name
        self.value = value  # If None, defaults to id (set later)
        self.initial = initial
        self.final = final
        self.transitions = StateTransitions()
        self._transitions_to = _TransitionBuilder(self)

    @property
    def to(self) -> _TransitionBuilder:
        return self._transitions_to

    def _clone(self) -> State:
        """Create a fresh copy without any transitions."""
        return State(
            name=self.name,
            value=self.value if self.value != self.id else None,
            initial=self.initial,
            final=self.final,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self.id == other.id and self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.id, self.name))

    def __repr__(self) -> str:
        flags = []
        if self.initial:
            flags.append("initial")
        if self.final:
            flags.append("final")
        extra = f" ({', '.join(flags)})" if flags else ""
        return f"State({self.name!r}, id={self.id!r}{extra})"
