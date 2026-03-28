"""StateMachine base class with metaclass for the custom FSM."""

from __future__ import annotations

from typing import Any

from agentspype.fsm.state import State, StateTransitions
from agentspype.fsm.transition import Transition, TransitionList


def _clone_state(state: State) -> State:
    """Create a fresh copy of a State object without any transitions."""
    return state._clone()


def _remap_transition_list(
    tl: TransitionList, state_map: dict[int, State]
) -> TransitionList:
    """Rebuild a TransitionList using fresh State copies.

    For each transition, remap source/target through state_map and create
    a new Transition that preserves all metadata (cond, unless).
    The new transition is registered on the new source state's internal
    transition list.
    """
    result = TransitionList()
    for t in tl.transitions:
        new_source = state_map.get(id(t.source), t.source)
        new_target = state_map.get(id(t.target), t.target)
        new_t = Transition(
            new_source,
            new_target,
            event=t.event,
            internal=t.internal,
            cond=list(t.cond),
            unless=list(t.unless),
        )
        new_tl = TransitionList([new_t])
        new_source.transitions.add_transitions(new_tl)
        result.add_transitions(new_tl)
    return result


def _clean_parent_mutations(
    old_transitions: list[Transition], state_map: dict[int, State]
) -> None:
    """Remove transitions that were added to parent State objects during class body execution.

    When a user writes ``ParentSM.idle.to(new_state)`` in a subclass body, the ``.to()``
    call mutates the parent's ``idle.transitions``. We undo this so the parent is not polluted.
    """
    for t in old_transitions:
        if id(t.source) in state_map:
            try:
                t.source.transitions._transitions.remove(t)
            except ValueError:
                pass
        if id(t.target) in state_map:
            try:
                t.target.transitions._transitions.remove(t)
            except ValueError:
                pass


def _collect_states_from_namespace(namespace: dict[str, Any]) -> list[State]:
    """Assign state.id from attribute names and collect all states."""
    all_states: list[State] = []
    for attr_name, attr_value in list(namespace.items()):
        if isinstance(attr_value, State):
            attr_value.id = attr_name
            if attr_value.value is None:
                attr_value.value = attr_name
            all_states.append(attr_value)
    return all_states


def _collect_transitions_from_namespace(
    namespace: dict[str, Any],
) -> tuple[dict[str, TransitionList], set[str]]:
    """Assign event names to transitions and collect all events."""
    all_transition_lists: dict[str, TransitionList] = {}
    all_event_names: set[str] = set()
    for attr_name, attr_value in list(namespace.items()):
        if isinstance(attr_value, TransitionList):
            all_transition_lists[attr_name] = attr_value
            all_event_names.add(attr_name)
            for t in attr_value.transitions:
                t.event = attr_name
    return all_transition_lists, all_event_names


def _build_transition_maps(
    all_transition_lists: dict[str, TransitionList],
) -> tuple[dict[tuple[str, str], list[Transition]], dict[str, list[Transition]]]:
    """Build transition lookup: (source_state_id, event) -> [Transition]."""
    transition_map: dict[tuple[str, str], list[Transition]] = {}
    event_map: dict[str, list[Transition]] = {}
    for event_name, tl in all_transition_lists.items():
        for t in tl.transitions:
            key = (t.source.id, event_name)
            transition_map.setdefault(key, []).append(t)
            event_map.setdefault(event_name, []).append(t)
    return transition_map, event_map


def _build_hook_tables(
    all_states: list[State],
    all_event_names: set[str],
    bases: tuple[type, ...],
    namespace: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Resolve hook methods by convention (on_enter_*, on_exit_*, on_*)."""
    enter_hooks: dict[str, str] = {}
    exit_hooks: dict[str, str] = {}
    event_hooks: dict[str, str] = {}

    # Collect method names from MRO + current namespace
    all_method_names: set[str] = set()
    for base in bases:
        for b in base.__mro__:
            all_method_names.update(
                k for k, v in vars(b).items() if callable(v) and not k.startswith("_")
            )
    all_method_names.update(
        k for k, v in namespace.items() if callable(v) and not k.startswith("_")
    )

    for state in all_states:
        hook_name = f"on_enter_{state.id}"
        if hook_name in all_method_names or hook_name in namespace:
            enter_hooks[state.id] = hook_name
        hook_name = f"on_exit_{state.id}"
        if hook_name in all_method_names or hook_name in namespace:
            exit_hooks[state.id] = hook_name

    for event_name in all_event_names:
        hook_name = f"on_{event_name}"
        if hook_name in all_method_names or hook_name in namespace:
            event_hooks[event_name] = hook_name

    return enter_hooks, exit_hooks, event_hooks


class _EventDescriptor:
    """Descriptor that maps ``self.event_name()`` to ``self.send("event_name")``."""

    __slots__ = ("event_name",)

    def __init__(self, event_name: str) -> None:
        self.event_name = event_name

    def __set_name__(self, owner: type, name: str) -> None:
        pass

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        evt = self.event_name

        def event_method(*, f: bool = False, **kwargs: Any) -> Any:
            return obj.send(evt, f=f, **kwargs)

        event_method.__name__ = evt
        return event_method


class StateMachineMeta(type):
    """Metaclass that:
    1. Collects State and TransitionList from class body + bases
    2. Clones inherited states (each subclass gets its own copies)
    3. Remaps transitions to reference the cloned states
    4. Assigns state.id from attribute name
    5. Builds the transition lookup table
    6. Resolves hook methods by convention
    """

    @staticmethod
    def _collect_parent_states(bases: tuple[type, ...]) -> dict[str, State]:
        """Collect all State objects inherited from parent classes."""
        parent_states: dict[str, State] = {}
        for base in bases:
            # Only access _states on already-processed FSM subclasses
            for state in getattr(base, "_states", []):
                if state.id and state.id not in parent_states:
                    parent_states[state.id] = state
        return parent_states

    @staticmethod
    def _clone_inherited_states(
        parent_states: dict[str, State], namespace: dict[str, Any]
    ) -> dict[int, State]:
        """Clone inherited states not redefined in the subclass namespace."""
        state_map: dict[int, State] = {}
        for state_id, old_state in parent_states.items():
            if state_id not in namespace or namespace[state_id] is old_state:
                new_state = _clone_state(old_state)
                namespace[state_id] = new_state
                state_map[id(old_state)] = new_state
        return state_map

    @staticmethod
    def _ensure_default_states(
        parent_states: dict[str, State], namespace: dict[str, Any]
    ) -> None:
        """Create default starting, idle, end states if missing."""
        if "starting" not in namespace and "starting" not in parent_states:
            namespace["starting"] = State("Starting", initial=True)
        if "idle" not in namespace and "idle" not in parent_states:
            namespace["idle"] = State("Idle")
        if "end" not in namespace and "end" not in parent_states:
            namespace["end"] = State("End", final=True)

    @staticmethod
    def _remap_namespace_transitions(
        state_map: dict[int, State], namespace: dict[str, Any]
    ) -> None:
        """Remap TransitionList values that reference parent states to use fresh clones."""
        if not state_map:
            return
        for key, value in list(namespace.items()):
            if isinstance(value, TransitionList):
                needs_remap = any(
                    id(t.source) in state_map or id(t.target) in state_map
                    for t in value.transitions
                )
                if needs_remap:
                    old_transitions = list(value.transitions)
                    namespace[key] = _remap_transition_list(value, state_map)
                    _clean_parent_mutations(old_transitions, state_map)

    @staticmethod
    def _remap_inherited_transitions(
        state_map: dict[int, State],
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> None:
        """Remap inherited transitions from parent classes using stored TransitionLists."""
        if not state_map:
            return
        seen: set[str] = set()
        for base in bases:
            parent_tls: dict[str, TransitionList] = getattr(
                base, "_transition_lists_", {}
            )
            for attr_name, tl in parent_tls.items():
                if attr_name in seen:
                    continue
                seen.add(attr_name)
                needs_remap = any(
                    id(t.source) in state_map or id(t.target) in state_map
                    for t in tl.transitions
                )
                if needs_remap:
                    remapped = _remap_transition_list(tl, state_map)
                    existing = namespace.get(attr_name)
                    if isinstance(existing, TransitionList):
                        namespace[attr_name] = existing | remapped
                    else:
                        namespace[attr_name] = remapped

    @staticmethod
    def _ensure_default_transitions(namespace: dict[str, Any]) -> None:
        """Create default start and stop transitions if missing."""
        if "start" not in namespace:
            namespace["start"] = namespace["starting"].to(namespace["idle"])
        if "stop" not in namespace:
            namespace["stop"] = namespace["starting"].to(namespace["end"]) | namespace[
                "idle"
            ].to(namespace["end"])

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        """Create a new StateMachine subclass with isolated state objects."""
        # Skip processing for the base StateMachine class itself
        is_base = not any(isinstance(b, StateMachineMeta) for b in bases)
        if is_base:
            return super().__new__(mcs, name, bases, namespace)

        # Phase 1: State and transition isolation
        parent_states = mcs._collect_parent_states(bases)
        state_map = mcs._clone_inherited_states(parent_states, namespace)
        mcs._ensure_default_states(parent_states, namespace)
        mcs._remap_namespace_transitions(state_map, namespace)
        mcs._remap_inherited_transitions(state_map, bases, namespace)
        mcs._ensure_default_transitions(namespace)

        # Phase 2: Collect and index
        all_states = _collect_states_from_namespace(namespace)
        all_transition_lists, all_event_names = _collect_transitions_from_namespace(
            namespace
        )
        namespace["_transition_lists_"] = dict(all_transition_lists)

        transition_map, event_map = _build_transition_maps(all_transition_lists)

        # Build state transitions info (unique_events per state)
        state_events: dict[str, set[str]] = {}
        for source_id, event_name in transition_map:
            state_events.setdefault(source_id, set()).add(event_name)
        for state in all_states:
            events = frozenset(state_events.get(state.id, set()))
            state.transitions = StateTransitions(
                transitions=state.transitions._transitions,
                events=events,
            )

        # Phase 3: Hooks, metadata, descriptors
        enter_hooks, exit_hooks, event_hooks = _build_hook_tables(
            all_states, all_event_names, bases, namespace
        )
        namespace.update(
            _states=all_states,
            _transition_map=transition_map,
            _event_map=event_map,
            _enter_hooks=enter_hooks,
            _exit_hooks=exit_hooks,
            _event_hooks=event_hooks,
            _all_event_names=all_event_names,
            _events_list=[_Event(n) for n in sorted(all_event_names)],
            _states_map={s.id: s for s in all_states},
        )
        for event_name in all_event_names:
            namespace[event_name] = _EventDescriptor(event_name)

        return super().__new__(mcs, name, bases, namespace)


class _Event:
    """Lightweight event descriptor for compatibility with ``sm.events``."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Event({self.name!r})"


class _StatesDescriptor:
    """Descriptor that works at both class and instance level for ``states``."""

    def __get__(self, obj: Any, objtype: type | None = None) -> list[State]:
        if objtype is None:
            objtype = type(obj)
        return list(getattr(objtype, "_states", []))


class _EventsDescriptor:
    """Descriptor that works at both class and instance level for ``events``."""

    def __get__(self, obj: Any, objtype: type | None = None) -> list[_Event]:
        if objtype is None:
            objtype = type(obj)
        return list(getattr(objtype, "_events_list", []))


class _StatesMapDescriptor:
    """Descriptor that works at both class and instance level for ``states_map``."""

    def __get__(self, obj: Any, objtype: type | None = None) -> dict[str, State]:
        if objtype is None:
            objtype = type(obj)
        return dict(getattr(objtype, "_states_map", {}))


class StateMachine(metaclass=StateMachineMeta):
    """Lightweight FSM base class.

    Provides the same API as ``python-statemachine.StateMachine`` for the
    subset of features used by agentspype.
    """

    # Class-level attributes set by metaclass
    _states: list[State]
    _transition_map: dict[tuple[str, str], list[Transition]]
    _event_map: dict[str, list[Transition]]
    _enter_hooks: dict[str, str]
    _exit_hooks: dict[str, str]
    _event_hooks: dict[str, str]
    _all_event_names: set[str]
    _events_list: list[_Event]
    _states_map: dict[str, State]
    _transition_lists_: dict[str, TransitionList]

    def __init__(self) -> None:
        # Find initial state
        initial = None
        for state in self._states:
            if state.initial:
                initial = state
                break
        if initial is None:
            raise ValueError(f"{self.__class__.__name__}: No initial state defined")
        self._current_state = initial

    # --- Event dispatch (hot path) ---

    def _check_guards(self, transition: Transition) -> bool:
        """Evaluate guard conditions. Returns True if transition is allowed."""
        if transition.cond:
            if not all(getattr(self, c)() for c in transition.cond):
                return False
        if transition.unless:
            if any(getattr(self, u)() for u in transition.unless):
                return False
        return True

    def _execute_transition(
        self, event: str, transition: Transition, **kwargs: Any
    ) -> None:
        """Execute a single transition: fire hooks and update state."""
        source = self._current_state
        target = transition.target

        self.before_transition(event, source.name, source.name, target.name)

        hook_name = self._event_hooks.get(event)
        if hook_name is not None:
            getattr(self, hook_name)(**kwargs)

        if not transition.internal:
            exit_hook_name = self._exit_hooks.get(source.id)
            if exit_hook_name is not None:
                getattr(self, exit_hook_name)()

            self._current_state = target

            enter_hook_name = self._enter_hooks.get(target.id)
            if enter_hook_name is not None:
                getattr(self, enter_hook_name)()

        self.after_transition(event, self._current_state)

    def send(self, event: str, *, f: bool = False, **kwargs: Any) -> Any:
        """Dispatch an event to the state machine.

        Args:
            event: The event name to fire.
            f: Force flag - bypasses guard conditions.
            **kwargs: Passed to ``on_<event>`` hooks.

        Returns:
            True if a transition was taken, False otherwise.
        """
        candidates = self._transition_map.get((self._current_state.id, event))
        if not candidates:
            if f:
                candidates = self._event_map.get(event)
            if not candidates:
                return False

        for transition in candidates:
            if not f and not self._check_guards(transition):
                continue
            self._execute_transition(event, transition, **kwargs)
            return True

        return False

    # --- Descriptors (work at both class and instance level) ---

    states = _StatesDescriptor()
    events = _EventsDescriptor()
    states_map = _StatesMapDescriptor()

    # --- Properties ---

    @property
    def current_state(self) -> State:
        return self._current_state

    # --- Hook methods (override in subclasses) ---

    def before_transition(
        self, event: str, state: str, source: str, target: str
    ) -> None:
        """Called before every transition. Override in subclasses."""

    def after_transition(self, event: str, state: Any) -> None:
        """Called after every transition. Override in subclasses."""
