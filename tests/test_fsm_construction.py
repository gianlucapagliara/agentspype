"""Unit tests for FSM construction: State, Transition, TransitionList, and metaclass."""

from __future__ import annotations

from agentspype.fsm import State, StateMachine, Transition, TransitionList
from agentspype.fsm.state import StateTransitions, _TransitionBuilder
from agentspype.fsm.transition import _normalize_guard

# ---------------------------------------------------------------------------
# TestStateConstruction
# ---------------------------------------------------------------------------


class TestStateConstruction:
    def test_defaults(self) -> None:
        s = State()
        assert s.name == ""
        assert s.value is None
        assert s.initial is False
        assert s.final is False
        assert s.id == ""
        assert isinstance(s.transitions, StateTransitions)
        assert len(s.transitions) == 0

    def test_all_params(self) -> None:
        s = State("Running", value="running", initial=True, final=False)
        assert s.name == "Running"
        assert s.value == "running"
        assert s.initial is True
        assert s.final is False

    def test_final_flag(self) -> None:
        s = State("Done", final=True)
        assert s.final is True
        assert s.initial is False

    def test_repr_no_flags(self) -> None:
        s = State("Active")
        s.id = "active"
        r = repr(s)
        assert "State(" in r
        assert "'Active'" in r
        assert "id='active'" in r
        assert "initial" not in r
        assert "final" not in r

    def test_repr_initial(self) -> None:
        s = State("Start", initial=True)
        s.id = "start"
        r = repr(s)
        assert "initial" in r

    def test_repr_final(self) -> None:
        s = State("End", final=True)
        s.id = "end"
        r = repr(s)
        assert "final" in r

    def test_repr_both_flags(self) -> None:
        s = State("Only", initial=True, final=True)
        s.id = "only"
        r = repr(s)
        assert "initial" in r
        assert "final" in r

    def test_to_property_returns_builder(self) -> None:
        s = State("X")
        assert isinstance(s.to, _TransitionBuilder)


# ---------------------------------------------------------------------------
# TestStateClone
# ---------------------------------------------------------------------------


class TestStateClone:
    def test_clone_preserves_name_initial_final(self) -> None:
        s = State("Idle", initial=True, final=False)
        s.id = "idle"
        s.value = "idle"
        clone = s._clone()
        assert clone.name == "Idle"
        assert clone.initial is True
        assert clone.final is False

    def test_clone_clears_transitions(self) -> None:
        a = State("A")
        b = State("B")
        a.id = "a"
        b.id = "b"
        a.to(b)
        assert len(a.transitions) == 1
        clone = a._clone()
        assert len(clone.transitions) == 0

    def test_clone_value_equals_id_resets_to_none(self) -> None:
        # When value == id the clone should pass value=None so metaclass can re-assign.
        s = State("Idle")
        s.id = "idle"
        s.value = "idle"  # value matches id
        clone = s._clone()
        # The clone's value should be None so the metaclass can reassign it from attr name.
        assert clone.value is None

    def test_clone_explicit_value_preserved(self) -> None:
        s = State("My State", value="custom_value")
        s.id = "my_state"
        clone = s._clone()
        assert clone.value == "custom_value"

    def test_clone_has_empty_id(self) -> None:
        s = State("Foo")
        s.id = "foo"
        clone = s._clone()
        assert clone.id == ""


# ---------------------------------------------------------------------------
# TestStateEquality
# ---------------------------------------------------------------------------


class TestStateEquality:
    def test_same_id_and_name_equal(self) -> None:
        a = State("Idle")
        a.id = "idle"
        b = State("Idle")
        b.id = "idle"
        assert a == b

    def test_different_id_not_equal(self) -> None:
        a = State("Idle")
        a.id = "idle"
        b = State("Idle")
        b.id = "other"
        assert a != b

    def test_different_name_not_equal(self) -> None:
        a = State("Idle")
        a.id = "idle"
        b = State("Busy")
        b.id = "idle"
        assert a != b

    def test_hash_consistency(self) -> None:
        a = State("Idle")
        a.id = "idle"
        b = State("Idle")
        b.id = "idle"
        assert hash(a) == hash(b)

    def test_comparison_with_non_state_returns_not_implemented(self) -> None:
        s = State("Idle")
        result = s.__eq__("not a state")
        assert result is NotImplemented

    def test_usable_in_set(self) -> None:
        a = State("Idle")
        a.id = "idle"
        b = State("Idle")
        b.id = "idle"
        s = {a, b}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# TestTransitionConstruction
# ---------------------------------------------------------------------------


class TestTransitionConstruction:
    def setup_method(self) -> None:
        self.src = State("A")
        self.src.id = "a"
        self.tgt = State("B")
        self.tgt.id = "b"

    def test_defaults(self) -> None:
        t = Transition(self.src, self.tgt)
        assert t.source is self.src
        assert t.target is self.tgt
        assert t.event == ""
        assert t.internal is False
        assert t.cond == []
        assert t.unless == []

    def test_all_params(self) -> None:
        t = Transition(
            self.src,
            self.tgt,
            event="go",
            internal=True,
            cond=["is_ready"],
            unless=["is_blocked"],
        )
        assert t.event == "go"
        assert t.internal is True
        assert t.cond == ["is_ready"]
        assert t.unless == ["is_blocked"]

    def test_cond_string_normalized(self) -> None:
        t = Transition(self.src, self.tgt, cond="is_ready")
        assert t.cond == ["is_ready"]

    def test_unless_tuple_normalized(self) -> None:
        t = Transition(self.src, self.tgt, unless=("blocked", "paused"))
        assert t.unless == ["blocked", "paused"]

    def test_repr(self) -> None:
        t = Transition(self.src, self.tgt, event="go")
        r = repr(t)
        assert "Transition(" in r
        assert "'a'" in r
        assert "'b'" in r
        assert "event='go'" in r


# ---------------------------------------------------------------------------
# TestNormalizeGuard
# ---------------------------------------------------------------------------


class TestNormalizeGuard:
    def test_none_returns_empty_list(self) -> None:
        assert _normalize_guard(None) == []

    def test_string_returns_list_with_string(self) -> None:
        assert _normalize_guard("is_ready") == ["is_ready"]

    def test_list_returned_as_is(self) -> None:
        result = _normalize_guard(["a", "b"])
        assert result == ["a", "b"]

    def test_tuple_converted_to_list(self) -> None:
        result = _normalize_guard(("x", "y"))
        assert result == ["x", "y"]
        assert isinstance(result, list)

    def test_empty_list(self) -> None:
        assert _normalize_guard([]) == []

    def test_empty_tuple(self) -> None:
        assert _normalize_guard(()) == []


# ---------------------------------------------------------------------------
# TestTransitionList
# ---------------------------------------------------------------------------


class TestTransitionList:
    def setup_method(self) -> None:
        self.a = State("A")
        self.a.id = "a"
        self.b = State("B")
        self.b.id = "b"
        self.c = State("C")
        self.c.id = "c"

    def test_init_empty(self) -> None:
        tl = TransitionList()
        assert tl.transitions == []

    def test_init_with_transitions(self) -> None:
        t = Transition(self.a, self.b)
        tl = TransitionList([t])
        assert len(tl.transitions) == 1
        assert tl.transitions[0] is t

    def test_or_combines(self) -> None:
        t1 = Transition(self.a, self.b)
        t2 = Transition(self.b, self.c)
        tl1 = TransitionList([t1])
        tl2 = TransitionList([t2])
        combined = tl1 | tl2
        assert len(combined.transitions) == 2
        assert t1 in combined.transitions
        assert t2 in combined.transitions

    def test_or_does_not_mutate_originals(self) -> None:
        t1 = Transition(self.a, self.b)
        t2 = Transition(self.b, self.c)
        tl1 = TransitionList([t1])
        tl2 = TransitionList([t2])
        _ = tl1 | tl2
        assert len(tl1.transitions) == 1
        assert len(tl2.transitions) == 1

    def test_add_transitions_extends(self) -> None:
        t1 = Transition(self.a, self.b)
        t2 = Transition(self.b, self.c)
        tl1 = TransitionList([t1])
        tl2 = TransitionList([t2])
        tl1.add_transitions(tl2)
        assert len(tl1.transitions) == 2

    def test_repr(self) -> None:
        tl = TransitionList()
        r = repr(tl)
        assert "TransitionList(" in r


# ---------------------------------------------------------------------------
# TestTransitionBuilder
# ---------------------------------------------------------------------------


class TestTransitionBuilder:
    def setup_method(self) -> None:
        self.src = State("Src")
        self.src.id = "src"
        self.tgt = State("Tgt")
        self.tgt.id = "tgt"

    def test_to_creates_transition_list(self) -> None:
        result = self.src.to(self.tgt)
        assert isinstance(result, TransitionList)
        assert len(result.transitions) == 1

    def test_to_registers_on_source_transitions(self) -> None:
        self.src.to(self.tgt)
        assert len(self.src.transitions) == 1
        assert self.src.transitions._transitions[0].target is self.tgt

    def test_to_itself_creates_self_transition(self) -> None:
        result = self.src.to.itself()
        assert isinstance(result, TransitionList)
        t = result.transitions[0]
        assert t.source is self.src
        assert t.target is self.src

    def test_to_with_cond(self) -> None:
        result = self.src.to(self.tgt, cond="is_ready")
        t = result.transitions[0]
        assert t.cond == ["is_ready"]

    def test_to_with_unless(self) -> None:
        result = self.src.to(self.tgt, unless="is_blocked")
        t = result.transitions[0]
        assert t.unless == ["is_blocked"]

    def test_to_with_internal(self) -> None:
        result = self.src.to(self.tgt, internal=True)
        t = result.transitions[0]
        assert t.internal is True

    def test_builder_source(self) -> None:
        builder = _TransitionBuilder(self.src)
        result = builder(self.tgt)
        assert result.transitions[0].source is self.src


# ---------------------------------------------------------------------------
# TestStateTransitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def setup_method(self) -> None:
        self.a = State("A")
        self.a.id = "a"
        self.b = State("B")
        self.b.id = "b"

    def test_init_defaults(self) -> None:
        st = StateTransitions()
        assert st._transitions == []
        assert st.unique_events == frozenset()

    def test_init_with_transitions_and_events(self) -> None:
        t = Transition(self.a, self.b, event="go")
        st = StateTransitions(transitions=[t], events=frozenset({"go"}))
        assert len(st) == 1
        assert st.unique_events == frozenset({"go"})

    def test_add_transitions(self) -> None:
        t = Transition(self.a, self.b, event="go")
        tl = TransitionList([t])
        st = StateTransitions()
        st.add_transitions(tl)
        assert len(st) == 1

    def test_iter(self) -> None:
        t = Transition(self.a, self.b, event="go")
        st = StateTransitions(transitions=[t])
        items = list(st)
        assert items == [t]

    def test_len(self) -> None:
        t1 = Transition(self.a, self.b, event="go")
        t2 = Transition(self.b, self.a, event="back")
        st = StateTransitions(transitions=[t1, t2])
        assert len(st) == 2

    def test_unique_events(self) -> None:
        st = StateTransitions(events=frozenset({"go", "stop"}))
        assert "go" in st.unique_events
        assert "stop" in st.unique_events


# ---------------------------------------------------------------------------
# TestMetaclassStateCollection
# ---------------------------------------------------------------------------


class TestMetaclassStateCollection:
    def test_state_id_assigned_from_attribute_name(self) -> None:
        # Provide all three default-required names to avoid extra states being injected.
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            running = State("Running")
            end = State("End", final=True)
            start = starting.to(running)
            stop = running.to(end)

        assert MySM.running.id == "running"

    def test_state_value_defaults_to_id_when_none(self) -> None:
        # Override all three defaults so no extra initial state is inserted.
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            active = State("Active")
            end = State("End", final=True)
            start = starting.to(active)
            stop = active.to(end)

        assert MySM.active.value == "active"

    def test_state_value_preserved_when_explicitly_set(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            active = State("Active", value="act")
            end = State("End", final=True)
            start = starting.to(active)
            stop = active.to(end)

        assert MySM.active.value == "act"

    def test_non_state_attributes_ignored(self) -> None:
        class MySM(StateMachine):
            my_number = 42
            my_string = "hello"

        state_ids = {s.id for s in MySM._states}
        assert "my_number" not in state_ids
        assert "my_string" not in state_ids

    def test_states_list_contains_all_declared_states(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            a = State("A")
            b = State("B")
            end = State("End", final=True)
            start = starting.to(a)
            go = a.to(b)
            stop = b.to(end)

        state_ids = {s.id for s in MySM._states}
        assert {"a", "b", "end", "starting"}.issubset(state_ids)


# ---------------------------------------------------------------------------
# TestMetaclassDefaultCreation
# ---------------------------------------------------------------------------


class TestMetaclassDefaultCreation:
    def test_default_states_created_when_missing(self) -> None:
        class MinimalSM(StateMachine):
            pass

        state_ids = {s.id for s in MinimalSM._states}
        assert "starting" in state_ids
        assert "idle" in state_ids
        assert "end" in state_ids

    def test_default_transitions_created(self) -> None:
        class MinimalSM(StateMachine):
            pass

        assert "start" in MinimalSM._all_event_names
        assert "stop" in MinimalSM._all_event_names

    def test_existing_states_not_overwritten(self) -> None:
        class MySM(StateMachine):
            starting = State("My Starting", initial=True)

        # The user's starting state should still be there
        assert MySM.starting.name == "My Starting"

    def test_existing_transitions_not_overwritten(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            s2 = State("S2")
            end = State("End", final=True)
            # Provide custom start transition (starting -> s2, not starting -> idle)
            start = starting.to(s2)
            stop = s2.to(end)

        # start transition should map starting -> s2, not starting -> idle
        start_transitions = MySM._transition_lists_["start"]
        targets = {t.target.id for t in start_transitions.transitions}
        assert "s2" in targets
        assert "idle" not in targets

    def test_starting_state_is_initial(self) -> None:
        class MinimalSM(StateMachine):
            pass

        initial_states = [s for s in MinimalSM._states if s.initial]
        assert len(initial_states) == 1
        assert initial_states[0].id == "starting"

    def test_end_state_is_final(self) -> None:
        class MinimalSM(StateMachine):
            pass

        end_state = MinimalSM._states_map["end"]
        assert end_state.final is True


# ---------------------------------------------------------------------------
# TestMetaclassTransitionMap
# ---------------------------------------------------------------------------


class TestMetaclassTransitionMap:
    def test_transition_map_keyed_by_state_event(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        assert ("starting", "go") in MySM._transition_map

    def test_event_map_keyed_by_event_name(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        assert "go" in MySM._event_map
        assert len(MySM._event_map["go"]) >= 1

    def test_state_transitions_unique_events_populated(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B")
            end = State("End", final=True)
            start = starting.to(b)
            go = starting.to(b)
            finish = b.to(end)
            stop = b.to(end)

        assert "go" in MySM.starting.transitions.unique_events
        assert "finish" in MySM.b.transitions.unique_events

    def test_transition_map_multiple_transitions_same_event(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B")
            end = State("End", final=True)
            start = starting.to(b)
            move = starting.to(b) | b.to(end)
            stop = b.to(end)

        assert ("starting", "move") in MySM._transition_map
        assert ("b", "move") in MySM._transition_map

    def test_transition_target_correct(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        transitions = MySM._transition_map[("starting", "go")]
        assert len(transitions) == 1
        assert transitions[0].target.id == "b"


# ---------------------------------------------------------------------------
# TestMetaclassEventDescriptors
# ---------------------------------------------------------------------------


class TestMetaclassEventDescriptors:
    def test_descriptors_created_for_all_events(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        assert "go" in MySM._all_event_names

    def test_descriptor_returns_callable_on_instance(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        sm = MySM()
        assert callable(sm.go)

    def test_descriptor_invokes_send(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)

        sm = MySM()
        assert sm.current_state.id == "starting"
        sm.start()
        assert sm.current_state.id == "b"

    def test_descriptor_returns_descriptor_on_class(self) -> None:
        from agentspype.fsm.machine import _EventDescriptor

        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        desc = MySM.__dict__["go"]
        assert isinstance(desc, _EventDescriptor)
        assert desc.event_name == "go"

    def test_descriptor_get_on_class_returns_descriptor(self) -> None:
        from agentspype.fsm.machine import _EventDescriptor

        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        result = MySM.go
        assert isinstance(result, _EventDescriptor)

    def test_multiple_events_all_have_descriptors(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B")
            end = State("End", final=True)
            start = starting.to(b)
            go = starting.to(b)
            finish = b.to(end)
            stop = b.to(end)

        sm = MySM()
        assert callable(sm.go)
        assert callable(sm.finish)

    def test_event_method_name_matches_event(self) -> None:
        class MySM(StateMachine):
            starting = State("Starting", initial=True)
            b = State("B", final=True)
            start = starting.to(b)
            go = starting.to(b)

        sm = MySM()
        method = sm.go
        assert method.__name__ == "go"
