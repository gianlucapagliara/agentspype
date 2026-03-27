import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from statemachine import State, StateMachine
from statemachine.factory import StateMachineMetaclass
from statemachine.transition import Transition
from statemachine.transition_list import TransitionList

from agentspype.agent.publishing import StateAgentPublishing

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent


T = TypeVar("T", bound="AgentStateMachine")


def _clone_state(state: State) -> State:
    """Create a fresh copy of a State object without any transitions."""
    return State(
        name=state.name,
        value=state.value if state.value != state.id else None,
        initial=state.initial,
        final=state.final,
    )


def _remap_transition_list(
    tl: TransitionList, state_map: dict[int, State]
) -> TransitionList:
    """Rebuild a TransitionList using fresh State copies.

    For each transition, remap source/target through state_map and create
    a new ``Transition`` that preserves all metadata (``cond``, ``unless``,
    ``validators``, ``before``, ``on``, ``after``) by copying the internal
    ``_specs`` callback list.  The new transition is registered on the new
    source state's internal transition list.
    """
    result = TransitionList()
    for t in tl.transitions:
        new_source = state_map.get(id(t.source), t.source)
        new_target = state_map.get(id(t.target), t.target)
        # Create a bare Transition then copy user-specified callback specs
        # (cond, unless, validators, before, on, after) from the original.
        # Convention callbacks (on_stop, before_transition, etc.) are NOT
        # copied — they will be added fresh by _setup() during class creation.
        new_t = Transition(new_source, new_target, internal=t.internal)
        for spec in t._specs.items:
            if not spec.is_convention:
                new_t._specs.items.append(spec)
        new_tl = TransitionList([new_t])
        # Register the transition on the source state
        new_source.transitions.add_transitions(new_tl)
        result.add_transitions(new_tl)
    return result


def _clean_parent_mutations(
    old_transitions: list[Transition], state_map: dict[int, State]
) -> None:
    """Remove transitions that were added to parent State objects during class body execution.

    When a user writes ``ParentSM.idle.to(new_state)`` in a subclass body, the ``.to()``
    call mutates the parent's ``idle.transitions``. We need to undo this mutation so the
    parent class is not polluted.

    This function iterates over each transition that was captured *before* remapping and,
    for every source/target that is about to be replaced by a fresh clone (i.e. present
    in ``state_map``), removes the stale transition from the original parent State's
    internal transition list. If the transition has already been removed (e.g. cleaned
    by a prior subclass definition), a warning is emitted instead of silently swallowing
    the error.
    """
    for t in old_transitions:
        if id(t.source) in state_map:
            try:
                t.source.transitions.transitions.remove(t)
            except ValueError:
                pass  # Already cleaned by a prior subclass or metaclass processing
        if id(t.target) in state_map:
            try:
                t.target.transitions.transitions.remove(t)
            except ValueError:
                pass  # Already cleaned by a prior subclass or metaclass processing


class AgentStateMachineMeta(StateMachineMetaclass):
    """Metaclass for AgentStateMachine that ensures proper state machine inheritance.

    This metaclass solves the state-sharing problem in python-statemachine's inheritance.
    When subclassing a concrete state machine, the parent's State objects would normally be
    shared (and mutated) across subclasses. This metaclass creates fresh State copies for
    each new class and remaps all transitions to reference the fresh copies.

    It also auto-creates the default ``starting``, ``idle``, ``end`` states and
    ``start``, ``stop`` transitions when they are not explicitly defined.
    """

    @staticmethod
    def _collect_parent_states(bases: tuple[type, ...]) -> dict[str, State]:
        """Collect all State objects inherited from parent classes."""
        parent_states: dict[str, State] = {}
        for base in bases:
            for state in getattr(base, "states", []):
                if state.id and state.id not in parent_states:
                    parent_states[state.id] = state
        return parent_states

    @staticmethod
    def _clone_inherited_states(
        parent_states: dict[str, State], namespace: dict[str, Any]
    ) -> dict[int, State]:
        """Clone inherited states not redefined in the subclass namespace.

        Returns a ``state_map`` mapping ``id(old_parent_state) -> new_fresh_state``.
        """
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
        """Create default ``starting``, ``idle``, ``end`` states if missing."""
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
        """Remap inherited transitions from parent classes using stored TransitionLists.

        After ``StateMachineMetaclass.__new__`` processes a class, all ``TransitionList``
        attributes are converted to bound methods, making them invisible to
        ``isinstance(..., TransitionList)`` checks.  To work around this, each class
        stores its original ``TransitionList`` objects in ``_transition_lists_``.

        This method walks the MRO, retrieves those stored TransitionLists, remaps them
        through ``state_map``, and either adds or **merges** them into the namespace.
        Merging is critical: when a subclass redefines a transition event (e.g. ``stop``),
        the parent's transitions for that event must be combined with the subclass's,
        not silently dropped.
        """
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
                        # Merge: combine subclass transitions with parent's
                        # remapped transitions so both sets of paths are kept.
                        namespace[attr_name] = existing | remapped
                    else:
                        namespace[attr_name] = remapped

    @staticmethod
    def _ensure_default_transitions(namespace: dict[str, Any]) -> None:
        """Create default ``start`` and ``stop`` transitions if missing."""
        if "start" not in namespace:
            namespace["start"] = namespace["starting"].to(namespace["idle"])
        if "stop" not in namespace:
            namespace["stop"] = namespace["starting"].to(namespace["end"]) | namespace[
                "idle"
            ].to(namespace["end"])

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        """Create a new AgentStateMachine subclass with isolated state objects."""
        if name == "AgentStateMachine":
            return super().__new__(mcs, name, bases, namespace)

        parent_states = mcs._collect_parent_states(bases)
        state_map = mcs._clone_inherited_states(parent_states, namespace)
        mcs._ensure_default_states(parent_states, namespace)
        mcs._remap_namespace_transitions(state_map, namespace)
        mcs._remap_inherited_transitions(state_map, bases, namespace)
        mcs._ensure_default_transitions(namespace)

        # Snapshot TransitionLists before super().__new__() converts them to
        # methods.  Subclasses use ``_transition_lists_`` to remap inherited
        # transitions (see ``_remap_inherited_transitions``).
        namespace["_transition_lists_"] = {
            key: value
            for key, value in namespace.items()
            if isinstance(value, TransitionList)
        }

        return super().__new__(mcs, name, bases, namespace)


class AgentStateMachine(StateMachine, metaclass=AgentStateMachineMeta):
    """Base class for all agent state machines.

    This class provides the basic state machine structure that all agents should inherit from.
    Child classes can override states and transitions by defining their own, or inherit the defaults.
    """

    # === States ===

    starting: ClassVar[State]
    idle: ClassVar[State]
    end: ClassVar[State]

    # === Transitions ===

    start: ClassVar[TransitionList]
    stop: ClassVar[TransitionList]

    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self._agent = weakref.ref(agent)
        self._strong_agent: Agent | None = None
        self._should_stop = False

    # === State actions ===

    def on_enter_end(self) -> None:
        # Pin a strong reference so the agent survives until the state-machine
        # transition fully completes (after_transition, etc.).  Without this,
        # Python 3.14's more aggressive GC can collect the agent as soon as
        # teardown() removes it from Agency's lists.
        self._strong_agent = self.agent
        self.agent.teardown()

    # === Transitions Actions ===

    def before_transition(
        self, event: str, state: str, source: str, target: str
    ) -> None:
        if source == target:
            return

        self.agent.logger().debug(
            f"[{self.agent.__class__.__name__}:StateMachine] ({event}) {source} -> {target}"
        )

    @abstractmethod
    def after_transition(self, event: str, state: State) -> None:
        """Handle behavior after a transition.

        This method should be implemented in child classes to handle post-transition behavior.
        Example implementation:
            self.agent.publishing.publish_transition(event, state)
        """
        raise NotImplementedError

    def on_start(self) -> None:
        self.agent.listening.subscribe()

    def on_stop(self) -> None:
        self._should_stop = True

    # === Conditions ===

    def should_stop(self) -> bool:
        return self._should_stop

    # === Properties ===

    @property
    def agent(self) -> "Agent":
        agent = self._agent()
        if agent is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: Agent reference expired (garbage collected or torn down)"
            )
        return agent

    # === Functions ===

    def safe_start(self) -> None:
        if not self.current_state.initial:
            return
        self.start(f=True)

    def safe_stop(self) -> None:
        """Safely stop the state machine if not already in final state."""
        if self.current_state.final:
            return
        self.stop(f=True)


# Example of how to create a concrete state machine:
class BasicAgentStateMachine(AgentStateMachine):
    """A basic implementation of an agent state machine.

    This class inherits all the default states and transitions from AgentStateMachine.
    It only needs to implement the required after_transition method.
    """

    def after_transition(self, event: str, state: State) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
