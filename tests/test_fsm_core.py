"""Comprehensive unit tests for the custom FSM engine (no Agent wrapper).

Tests the core StateMachine, State, Transition, and TransitionList classes
directly via:
    from agentspype.fsm import State, StateMachine, Transition, TransitionList

Design note
-----------
The metaclass (StateMachineMeta) auto-creates ``starting`` (initial=True),
``idle``, and ``end`` (final=True) unless they are explicitly defined in the
class body.  Inline SM subclasses that declare their own states must therefore
also redeclare ``starting`` / ``idle`` / ``end`` to suppress the auto-created
ones, otherwise two initial states exist and class creation raises ValueError.

The simplest strategy is to use the auto-created defaults as the state names
everywhere.  For tests that need distinct named states we simply override
the three defaults with non-conflicting flags.
"""

from __future__ import annotations

import pytest

from agentspype.fsm import State, StateMachine, Transition, TransitionList

# ---------------------------------------------------------------------------
# Module-level SM fixtures (avoids recreating classes inside every test method)
# ---------------------------------------------------------------------------


class _SimpleSM(StateMachine):
    """starting(initial) -> idle -> end(final)  using auto-defaults."""

    # Uses auto-created starting / idle / end + auto start / stop transitions.


class _ThreeStateSM(StateMachine):
    """Custom three-state machine.  Overrides defaults to avoid conflict."""

    starting = State("Starting", initial=True)
    idle = State("Idle")
    end = State("End", final=True)
    go = starting.to(idle)
    stop = idle.to(end)


# ---------------------------------------------------------------------------
# Helper: fresh instance factories
# ---------------------------------------------------------------------------


def _simple() -> StateMachine:
    return _SimpleSM()


def _three() -> StateMachine:
    return _ThreeStateSM()


# ===========================================================================
# 1. TestStateMachineInit
# ===========================================================================


class TestStateMachineInit:
    """Initial state setup and validation at class-creation time."""

    def test_initial_state_set_on_construction(self) -> None:
        sm = _simple()
        assert sm.current_state.id == "starting"

    def test_no_initial_state_raises_on_instantiation(self) -> None:
        class NoInitialSM(StateMachine):
            starting = State("Starting")  # initial=False — suppresses auto-initial
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

        with pytest.raises(ValueError, match="No initial state"):
            NoInitialSM()

    def test_multiple_initial_states_raises_at_class_creation(self) -> None:
        with pytest.raises(ValueError, match="Multiple initial states"):

            class TwoInitialsSM(StateMachine):
                # 'starting' auto-creates with initial=True; adding another
                # initial state triggers the validation error.
                extra_initial = State("Extra", initial=True)
                end = State("End", final=True)
                go = extra_initial.to(end)
                # suppress the default stop so we don't get a missing-event error
                stop = extra_initial.to(end)

    def test_custom_value_preserved(self) -> None:
        class CustomValueSM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle", value="my_idle_value")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end)

        assert CustomValueSM.idle.value == "my_idle_value"

    def test_auto_created_default_states_present(self) -> None:
        class MinimalSM(StateMachine):
            pass

        state_ids = {s.id for s in MinimalSM.states}
        assert {"starting", "idle", "end"} <= state_ids

    def test_auto_created_starting_is_initial(self) -> None:
        class MinimalSM(StateMachine):
            pass

        assert MinimalSM.starting.initial is True

    def test_auto_created_end_is_final(self) -> None:
        class MinimalSM(StateMachine):
            pass

        assert MinimalSM.end.final is True

    def test_current_state_is_initial_state(self) -> None:
        sm = _simple()
        assert sm.current_state.initial is True

    def test_state_value_defaults_to_id_when_none_given(self) -> None:
        """When value= is omitted, State.value defaults to the attribute name (id)."""
        assert _SimpleSM.idle.value == "idle"


# ===========================================================================
# 2. TestEventDispatch
# ===========================================================================


class TestEventDispatch:
    """send() semantics: valid/invalid events, state updates, descriptors."""

    def test_send_valid_event_returns_true(self) -> None:
        sm = _three()
        assert sm.send("go") is True

    def test_send_invalid_event_returns_false(self) -> None:
        sm = _three()
        assert sm.send("nonexistent_event") is False

    def test_send_wrong_state_returns_false(self) -> None:
        sm = _three()
        # "stop" is only valid from idle, not starting
        assert sm.send("stop") is False

    def test_send_updates_current_state(self) -> None:
        sm = _three()
        sm.send("go")
        assert sm.current_state.id == "idle"

    def test_send_sequence_updates_state_correctly(self) -> None:
        sm = _three()
        sm.send("go")
        sm.send("stop")
        assert sm.current_state.id == "end"

    def test_event_descriptor_calls_send(self) -> None:
        sm = _three()
        result = sm.go()
        assert result is True
        assert sm.current_state.id == "idle"

    def test_event_descriptor_passes_force_flag(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond="always_false")
            stop = idle.to(end)

            def always_false(self) -> bool:
                return False

        sm = SM()
        assert sm.go() is False
        assert sm.go(f=True) is True
        assert sm.current_state.id == "idle"

    def test_send_passes_kwargs_to_hook(self) -> None:
        received: dict = {}

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_go(self, **kwargs: object) -> None:
                received.update(kwargs)

        sm = SM()
        sm.send("go", x=42, y="hello")
        assert received == {"x": 42, "y": "hello"}

    def test_event_descriptor_passes_kwargs(self) -> None:
        received: dict = {}

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_go(self, **kwargs: object) -> None:
                received.update(kwargs)

        sm = SM()
        sm.go(foo="bar")
        assert received == {"foo": "bar"}


# ===========================================================================
# 3. TestFinalState
# ===========================================================================


class TestFinalState:
    """Final state blocks all further events."""

    def _reach_final(self, sm: StateMachine) -> None:
        sm.send("go")
        sm.send("stop")

    def test_send_from_final_returns_false(self) -> None:
        sm = _three()
        self._reach_final(sm)
        assert sm.send("go") is False

    def test_send_from_final_with_force_flag_returns_false(self) -> None:
        sm = _three()
        self._reach_final(sm)
        assert sm.send("go", f=True) is False

    def test_final_state_property(self) -> None:
        sm = _three()
        self._reach_final(sm)
        assert sm.current_state.final is True

    def test_non_final_state_property(self) -> None:
        sm = _three()
        assert sm.current_state.final is False

    def test_only_final_state_blocks_not_regular(self) -> None:
        sm = _three()
        sm.send("go")
        assert sm.current_state.final is False
        assert sm.send("stop") is True


# ===========================================================================
# 4. TestGuardConditions
# ===========================================================================


class TestGuardConditions:
    """cond=, unless=, combined, force bypass."""

    def test_cond_true_allows_transition(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond="is_allowed")
            stop = idle.to(end)

            def is_allowed(self) -> bool:
                return True

        sm = SM()
        assert sm.send("go") is True
        assert sm.current_state.id == "idle"

    def test_cond_false_blocks_transition(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond="is_allowed")
            stop = idle.to(end)

            def is_allowed(self) -> bool:
                return False

        sm = SM()
        assert sm.send("go") is False
        assert sm.current_state.id == "starting"

    def test_unless_true_blocks_transition(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, unless="is_blocked")
            stop = idle.to(end)

            def is_blocked(self) -> bool:
                return True

        sm = SM()
        assert sm.send("go") is False

    def test_unless_false_allows_transition(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, unless="is_blocked")
            stop = idle.to(end)

            def is_blocked(self) -> bool:
                return False

        sm = SM()
        assert sm.send("go") is True

    def test_multiple_conds_all_must_pass(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond=["cond_a", "cond_b"])
            stop = idle.to(end)

            def cond_a(self) -> bool:
                return True

            def cond_b(self) -> bool:
                return False

        sm = SM()
        assert sm.send("go") is False

    def test_multiple_conds_all_true_passes(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond=["cond_a", "cond_b"])
            stop = idle.to(end)

            def cond_a(self) -> bool:
                return True

            def cond_b(self) -> bool:
                return True

        sm = SM()
        assert sm.send("go") is True

    def test_multiple_unless_any_blocks(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, unless=["block_a", "block_b"])
            stop = idle.to(end)

            def block_a(self) -> bool:
                return False

            def block_b(self) -> bool:
                return True

        sm = SM()
        assert sm.send("go") is False

    def test_combined_cond_and_unless(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond="is_ok", unless="is_blocked")
            stop = idle.to(end)

            def is_ok(self) -> bool:
                return True

            def is_blocked(self) -> bool:
                return True

        sm = SM()
        assert sm.send("go") is False

    def test_force_bypasses_cond(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond="always_false")
            stop = idle.to(end)

            def always_false(self) -> bool:
                return False

        sm = SM()
        assert sm.send("go", f=True) is True
        assert sm.current_state.id == "idle"

    def test_force_bypasses_unless(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, unless="always_true")
            stop = idle.to(end)

            def always_true(self) -> bool:
                return True

        sm = SM()
        assert sm.send("go", f=True) is True
        assert sm.current_state.id == "idle"

    def test_force_bypasses_source_state_check(self) -> None:
        """With f=True, _event_map is used so wrong source state is ignored."""

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            processing = State("Processing")
            end = State("End", final=True)
            go = starting.to(idle)
            # "finish" only defined from idle -> processing
            finish = idle.to(processing)
            stop = processing.to(end)

        sm = SM()
        # Currently in starting; "finish" normally only fires from idle
        assert sm.send("finish") is False
        # Force bypasses source check
        assert sm.send("finish", f=True) is True
        assert sm.current_state.id == "processing"


# ===========================================================================
# 5. TestInternalTransitions
# ===========================================================================


class TestInternalTransitions:
    """internal=True: no state change, no enter/exit hooks, event hook fires."""

    def test_internal_transition_does_not_change_state(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            ping = starting.to(starting, internal=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        sm.send("ping")
        assert sm.current_state.id == "starting"

    def test_internal_transition_does_not_fire_exit_hook(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            ping = starting.to(starting, internal=True)
            start = starting.to(idle)
            stop = idle.to(end)

            def on_exit_starting(self) -> None:
                log.append("exit_starting")

        sm = SM()
        sm.send("ping")
        assert "exit_starting" not in log

    def test_internal_transition_does_not_fire_enter_hook(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            ping = starting.to(starting, internal=True)
            start = starting.to(idle)
            stop = idle.to(end)

            def on_enter_starting(self) -> None:
                log.append("enter_starting")

        sm = SM()
        sm.send("ping")
        assert "enter_starting" not in log

    def test_internal_transition_fires_event_hook(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            ping = starting.to(starting, internal=True)
            start = starting.to(idle)
            stop = idle.to(end)

            def on_ping(self) -> None:
                log.append("on_ping")

        sm = SM()
        sm.send("ping")
        assert "on_ping" in log

    def test_external_self_transition_fires_enter_exit(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            # external self-loop (internal=False by default)
            loop = starting.to(starting)
            start = starting.to(idle)
            stop = idle.to(end)

            def on_exit_starting(self) -> None:
                log.append("exit_starting")

            def on_enter_starting(self) -> None:
                log.append("enter_starting")

        sm = SM()
        sm.send("loop")
        assert "exit_starting" in log
        assert "enter_starting" in log

    def test_internal_returns_true(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            ping = starting.to(starting, internal=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        assert sm.send("ping") is True


# ===========================================================================
# 6. TestHooks
# ===========================================================================


class TestHooks:
    """Hook firing, kwargs passing, parent resolution, and execution order."""

    def test_on_enter_fires_on_state_entry(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_enter_idle(self) -> None:
                log.append("enter_idle")

        sm = SM()
        sm.send("go")
        assert "enter_idle" in log

    def test_on_exit_fires_on_state_exit(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_exit_starting(self) -> None:
                log.append("exit_starting")

        sm = SM()
        sm.send("go")
        assert "exit_starting" in log

    def test_on_event_fires_with_kwargs(self) -> None:
        received: dict = {}

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_go(self, **kwargs: object) -> None:
                received.update(kwargs)

        sm = SM()
        sm.send("go", value=99)
        assert received == {"value": 99}

    def test_before_event_fires(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def before_go(self) -> None:
                log.append("before_go")

        sm = SM()
        sm.send("go")
        assert "before_go" in log

    def test_after_event_fires(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def after_go(self) -> None:
                log.append("after_go")

        sm = SM()
        sm.send("go")
        assert "after_go" in log

    def test_hook_execution_order(self) -> None:
        """Exact hook order: before_transition → before_<e> → on_<e> →
        on_exit_<src> → on_enter_<tgt> → after_<e> → after_transition."""
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def before_transition(
                self, event: str, state: str, source: str, target: str
            ) -> None:
                log.append("before_transition")

            def before_go(self) -> None:
                log.append("before_go")

            def on_go(self) -> None:
                log.append("on_go")

            def on_exit_starting(self) -> None:
                log.append("on_exit_starting")

            def on_enter_idle(self) -> None:
                log.append("on_enter_idle")

            def after_go(self) -> None:
                log.append("after_go")

            def after_transition(self, event: str, state: object) -> None:
                log.append("after_transition")

        sm = SM()
        sm.send("go")
        assert log == [
            "before_transition",
            "before_go",
            "on_go",
            "on_exit_starting",
            "on_enter_idle",
            "after_go",
            "after_transition",
        ]

    def test_hooks_from_parent_class_are_resolved(self) -> None:
        log: list[str] = []

        class BaseSM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_go(self) -> None:
                log.append("parent_on_go")

        class ChildSM(BaseSM):
            pass

        sm = ChildSM()
        sm.send("go")
        assert "parent_on_go" in log

    def test_before_event_hook_receives_kwargs(self) -> None:
        received: dict = {}

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def before_go(self, **kwargs: object) -> None:
                received.update(kwargs)

        sm = SM()
        sm.send("go", token="abc")
        assert received == {"token": "abc"}

    def test_after_event_hook_receives_kwargs(self) -> None:
        received: dict = {}

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def after_go(self, **kwargs: object) -> None:
                received.update(kwargs)

        sm = SM()
        sm.send("go", result=True)
        assert received == {"result": True}

    def test_before_transition_global_hook_receives_event_and_states(self) -> None:
        calls: list[tuple] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def before_transition(
                self, event: str, state: str, source: str, target: str
            ) -> None:
                calls.append((event, state, source, target))

        sm = SM()
        sm.send("go")
        assert len(calls) == 1
        event, state, source, target = calls[0]
        assert event == "go"
        # before_transition receives state.name (display name), not state.id
        assert source == "Starting"
        assert target == "Idle"

    def test_after_transition_global_hook_receives_current_state(self) -> None:
        calls: list = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def after_transition(self, event: str, state: object) -> None:
                calls.append(state)

        sm = SM()
        sm.send("go")
        assert len(calls) == 1
        assert calls[0].id == "idle"  # type: ignore[union-attr]


# ===========================================================================
# 7. TestSelfTransitions
# ===========================================================================


class TestSelfTransitions:
    """to.itself(), internal vs. external self-transitions."""

    def test_to_itself_stays_in_same_state(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            loop = starting.to.itself()
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        sm.send("loop")
        assert sm.current_state.id == "starting"

    def test_internal_self_no_enter_exit(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            loop = starting.to.itself(internal=True)
            start = starting.to(idle)
            stop = idle.to(end)

            def on_enter_starting(self) -> None:
                log.append("enter")

            def on_exit_starting(self) -> None:
                log.append("exit")

        sm = SM()
        sm.send("loop")
        assert "enter" not in log
        assert "exit" not in log

    def test_external_self_fires_enter_exit(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            loop = starting.to.itself()  # external (internal=False by default)
            start = starting.to(idle)
            stop = idle.to(end)

            def on_enter_starting(self) -> None:
                log.append("enter")

            def on_exit_starting(self) -> None:
                log.append("exit")

        sm = SM()
        sm.send("loop")
        assert "exit" in log
        assert "enter" in log

    def test_itself_with_cond_blocks_when_false(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            loop = starting.to.itself(cond="is_ok")
            start = starting.to(idle)
            stop = idle.to(end)
            _ok = True

            def is_ok(self) -> bool:
                return self._ok  # type: ignore[attr-defined]

        sm = SM()
        sm._ok = False  # type: ignore[attr-defined]
        assert sm.send("loop") is False

    def test_itself_with_cond_passes_when_true(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            end = State("End", final=True)
            idle = State("Idle")
            loop = starting.to.itself(cond="is_ok")
            start = starting.to(idle)
            stop = idle.to(end)

            def is_ok(self) -> bool:
                return True

        sm = SM()
        assert sm.send("loop") is True
        assert sm.current_state.id == "starting"


# ===========================================================================
# 8. TestAccessors
# ===========================================================================


class TestAccessors:
    """states, events, states_map at class and instance level."""

    def test_states_at_class_level(self) -> None:
        states = _ThreeStateSM.states
        assert isinstance(states, list)
        ids = {s.id for s in states}
        assert {"starting", "idle", "end"} == ids

    def test_states_at_instance_level(self) -> None:
        sm = _three()
        states = sm.states
        assert isinstance(states, list)
        ids = {s.id for s in states}
        assert {"starting", "idle", "end"} == ids

    def test_events_at_class_level(self) -> None:
        events = _ThreeStateSM.events
        names = {e.name for e in events}
        assert "go" in names
        assert "stop" in names

    def test_events_at_instance_level(self) -> None:
        sm = _three()
        events = sm.events
        names = {e.name for e in events}
        assert "go" in names
        assert "stop" in names

    def test_states_map_at_class_level(self) -> None:
        smap = _ThreeStateSM.states_map
        assert isinstance(smap, dict)
        assert "starting" in smap
        assert smap["starting"].id == "starting"

    def test_states_map_at_instance_level(self) -> None:
        sm = _three()
        smap = sm.states_map
        assert isinstance(smap, dict)
        assert "idle" in smap

    def test_current_state_property(self) -> None:
        sm = _three()
        state = sm.current_state
        assert isinstance(state, State)
        assert state.id == "starting"

    def test_states_returns_copy_not_original(self) -> None:
        """Mutating the returned list must not affect the class."""
        states1 = _ThreeStateSM.states
        states1.clear()
        assert len(_ThreeStateSM.states) > 0

    def test_states_map_returns_copy(self) -> None:
        smap = _ThreeStateSM.states_map
        smap.clear()
        assert len(_ThreeStateSM.states_map) > 0

    def test_events_include_auto_default_events(self) -> None:
        class MinimalSM(StateMachine):
            pass

        names = {e.name for e in MinimalSM.events}
        assert "start" in names
        assert "stop" in names

    def test_states_map_keys_match_state_ids(self) -> None:
        smap = _ThreeStateSM.states_map
        for key, state in smap.items():
            assert key == state.id


# ===========================================================================
# 9. TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Multiple transitions, guard exceptions, hook exceptions, empty SM."""

    def test_multiple_transitions_same_event_first_valid_wins_guard_passes(
        self,
    ) -> None:
        """When guard on first candidate passes, it is taken."""

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            processing = State("Processing")
            end = State("End", final=True)
            go = starting.to(idle, cond="use_idle") | starting.to(processing)
            stop = idle.to(end) | processing.to(end)

            def use_idle(self) -> bool:
                return True

        sm = SM()
        sm.send("go")
        assert sm.current_state.id == "idle"

    def test_multiple_transitions_same_event_falls_through_to_second(self) -> None:
        """When guard on first candidate fails, second (open) candidate is taken."""

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            processing = State("Processing")
            end = State("End", final=True)
            go = starting.to(idle, cond="use_idle") | starting.to(processing)
            stop = idle.to(end) | processing.to(end)

            def use_idle(self) -> bool:
                return False

        sm = SM()
        sm.send("go")
        assert sm.current_state.id == "processing"

    def test_guard_exception_propagates(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle, cond="bad_guard")
            stop = idle.to(end)

            def bad_guard(self) -> bool:
                raise RuntimeError("guard blew up")

        sm = SM()
        with pytest.raises(RuntimeError, match="guard blew up"):
            sm.send("go")

    def test_hook_exception_propagates(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

            def on_go(self) -> None:
                raise ValueError("hook blew up")

        sm = SM()
        with pytest.raises(ValueError, match="hook blew up"):
            sm.send("go")

    def test_empty_sm_only_has_defaults(self) -> None:
        class EmptySM(StateMachine):
            pass

        state_ids = {s.id for s in EmptySM.states}
        assert state_ids == {"starting", "idle", "end"}
        event_names = {e.name for e in EmptySM.events}
        assert event_names == {"start", "stop"}

    def test_send_never_raises_for_unknown_event(self) -> None:
        sm = _three()
        result = sm.send("totally_unknown_event_xyz")
        assert result is False

    def test_transition_list_pipe_operator(self) -> None:
        """TransitionList | TransitionList produces merged list."""
        s1 = State("S1", initial=True)
        s2 = State("S2")
        s3 = State("S3")
        t1 = Transition(s1, s2)
        t2 = Transition(s1, s3)
        tl1 = TransitionList([t1])
        tl2 = TransitionList([t2])
        merged = tl1 | tl2
        assert len(merged.transitions) == 2
        assert t1 in merged.transitions
        assert t2 in merged.transitions

    def test_two_instances_are_independent(self) -> None:
        """Two instances of the same SM class must not share state."""
        sm1 = _three()
        sm2 = _three()
        sm1.send("go")
        assert sm1.current_state.id == "idle"
        assert sm2.current_state.id == "starting"

    def test_state_value_defaults_to_id(self) -> None:
        """When value= is not given, State.value defaults to attribute name."""
        assert _ThreeStateSM.starting.value == "starting"
        assert _ThreeStateSM.idle.value == "idle"
        assert _ThreeStateSM.end.value == "end"

    def test_state_repr_includes_flags(self) -> None:
        s = State("Test", initial=True, final=False)
        s.id = "test"
        r = repr(s)
        assert "initial" in r

    def test_transition_repr(self) -> None:
        s1 = State("S1")
        s1.id = "s1"
        s2 = State("S2")
        s2.id = "s2"
        t = Transition(s1, s2, event="go")
        r = repr(t)
        assert "s1" in r
        assert "s2" in r
        assert "go" in r

    def test_no_initial_but_with_final_raises_on_instantiation(self) -> None:
        """Confirms ValueError message says 'No initial state'."""

        class SM(StateMachine):
            starting = State("NotInitial")  # suppress auto-initial
            idle = State("Idle")
            end = State("End", final=True)
            go = starting.to(idle)
            stop = idle.to(end)

        with pytest.raises(ValueError, match="No initial state"):
            SM()
