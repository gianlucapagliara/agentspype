"""Tests for Run-To-Completion (RTC) event queue semantics in StateMachine."""

from __future__ import annotations

import pytest

from agentspype.fsm import State, StateMachine

# ---------------------------------------------------------------------------
# Helper state machines
# ---------------------------------------------------------------------------


class ReentrantOnEnterMachine(StateMachine):
    """on_enter_idle sends 'activate' — mirrors ClockStateMachine pattern."""

    starting = State("Starting", initial=True)
    idle = State("Idle")
    running = State("Running")
    end = State("End", final=True)

    start = starting.to(idle)
    activate = idle.to(running)
    stop = running.to(end) | idle.to(end)

    def __init__(self) -> None:
        super().__init__()
        self.hook_log: list[str] = []

    def on_enter_idle(self) -> None:
        self.hook_log.append(f"on_enter_idle:state={self._current_state.id}")
        self.send("activate")

    def after_start(self) -> None:
        self.hook_log.append(f"after_start:state={self._current_state.id}")

    def after_activate(self) -> None:
        self.hook_log.append(f"after_activate:state={self._current_state.id}")

    def after_transition(self, event: str, state: object) -> None:
        self.hook_log.append(
            f"after_transition({event}):state={self._current_state.id}"
        )


# ---------------------------------------------------------------------------
# Core RTC semantics
# ---------------------------------------------------------------------------


class TestRTCBasicSemantics:
    """Events sent from hooks are deferred, not nested."""

    def test_send_during_on_enter_is_deferred(self) -> None:
        sm = ReentrantOnEnterMachine()
        sm.send("start")
        assert sm.current_state.id == "running"

    def test_after_hooks_see_correct_state(self) -> None:
        """The core bug fix: after_start sees 'idle', not 'running'."""
        sm = ReentrantOnEnterMachine()
        sm.send("start")

        # after_start should see idle (its own transition's target)
        assert "after_start:state=idle" in sm.hook_log
        # after_activate should see running
        assert "after_activate:state=running" in sm.hook_log

    def test_after_transition_sees_correct_state_per_event(self) -> None:
        sm = ReentrantOnEnterMachine()
        sm.send("start")

        # after_transition for "start" should see idle
        assert "after_transition(start):state=idle" in sm.hook_log
        # after_transition for "activate" should see running
        assert "after_transition(activate):state=running" in sm.hook_log

    def test_hook_execution_order_with_rtc(self) -> None:
        sm = ReentrantOnEnterMachine()
        sm.send("start")

        expected = [
            "on_enter_idle:state=idle",
            "after_start:state=idle",
            "after_transition(start):state=idle",
            # Queue drains: activate fires now
            "after_activate:state=running",
            "after_transition(activate):state=running",
        ]
        assert sm.hook_log == expected

    def test_send_during_on_exit_is_deferred(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            done = State("Done")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(done)
            finish = done.to(end)
            stop = idle.to(end) | done.to(end)

            def on_exit_idle(self) -> None:
                log.append(f"on_exit_idle:state={self._current_state.id}")
                self.send("finish")

            def after_go(self) -> None:
                log.append(f"after_go:state={self._current_state.id}")

        sm = SM()
        sm.send("start")
        sm.send("go")
        # after_go should see 'done' (not 'end')
        assert "after_go:state=done" in log
        assert sm.current_state.id == "end"

    def test_send_during_before_event_is_deferred(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            finish = active.to(end)
            stop = idle.to(end) | active.to(end)

            def before_go(self) -> None:
                self.send("finish")

            def after_go(self) -> None:
                log.append(f"after_go:state={self._current_state.id}")

        sm = SM()
        sm.send("start")
        sm.send("go")
        # after_go sees 'active', then finish drains from queue
        assert "after_go:state=active" in log
        assert sm.current_state.id == "end"

    def test_send_during_on_event_is_deferred(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            finish = active.to(end)
            stop = idle.to(end) | active.to(end)

            def on_go(self) -> None:
                self.send("finish")

            def after_go(self) -> None:
                log.append(f"after_go:state={self._current_state.id}")

        sm = SM()
        sm.send("start")
        sm.send("go")
        assert "after_go:state=active" in log
        assert sm.current_state.id == "end"

    def test_send_during_after_event_is_deferred(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            finish = active.to(end)
            stop = idle.to(end) | active.to(end)

            def after_go(self) -> None:
                self.send("finish")

        sm = SM()
        sm.send("start")
        sm.send("go")
        assert sm.current_state.id == "end"


# ---------------------------------------------------------------------------
# Queue ordering
# ---------------------------------------------------------------------------


class TestQueueOrdering:
    def test_multiple_queued_events_execute_fifo(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            a = State("A")
            b = State("B")
            end = State("End", final=True)

            start = starting.to(idle)
            go_a = idle.to(a)
            go_b = a.to(b)
            finish = b.to(end)
            stop = idle.to(end) | a.to(end) | b.to(end)

            def on_enter_idle(self) -> None:
                log.append("queue_go_a")
                self.send("go_a")
                log.append("queue_go_b")
                self.send("go_b")

        sm = SM()
        sm.send("start")
        # go_a should fire first, then go_b
        assert sm.current_state.id == "b"
        assert log == ["queue_go_a", "queue_go_b"]

    def test_chained_queuing(self) -> None:
        """Event A queues B, B's hook queues C."""
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            a = State("A")
            b = State("B")
            end = State("End", final=True)

            start = starting.to(idle)
            go_a = idle.to(a)
            go_b = a.to(b)
            finish = b.to(end)
            stop = idle.to(end) | a.to(end) | b.to(end)

            def on_enter_idle(self) -> None:
                log.append("enter_idle")
                self.send("go_a")

            def on_enter_a(self) -> None:
                log.append("enter_a")
                self.send("go_b")

            def on_enter_b(self) -> None:
                log.append("enter_b")
                self.send("finish")

        sm = SM()
        sm.send("start")
        assert sm.current_state.id == "end"
        assert log == ["enter_idle", "enter_a", "enter_b"]


# ---------------------------------------------------------------------------
# Return values
# ---------------------------------------------------------------------------


class TestReturnValues:
    def test_queued_event_returns_true(self) -> None:
        """send() from within a hook returns True (queued for execution)."""
        return_values: list[bool] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            running = State("Running")
            end = State("End", final=True)

            start = starting.to(idle)
            activate = idle.to(running)
            stop = running.to(end) | idle.to(end)

            def on_enter_idle(self) -> None:
                result = self.send("activate")
                return_values.append(result)

        sm = SM()
        sm.send("start")
        assert return_values == [True]

    def test_direct_send_returns_true_on_success(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        assert sm.send("start") is True

    def test_direct_send_returns_false_on_no_transition(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        assert sm.send("nonexistent") is False

    def test_send_returns_false_in_final_state(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        sm.send("start")
        sm.send("stop")
        assert sm.send("start") is False


# ---------------------------------------------------------------------------
# Final state handling
# ---------------------------------------------------------------------------


class TestFinalStateHandling:
    def test_queue_cleared_on_final_state(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)

            start = starting.to(idle)
            stop = idle.to(end)

            def on_enter_idle(self) -> None:
                self.send("stop")
                # This event should be discarded — machine will be in final state
                self.send("start")

            def on_enter_end(self) -> None:
                log.append("entered_end")

        sm = SM()
        sm.send("start")
        assert sm.current_state.id == "end"
        assert log == ["entered_end"]

    def test_queued_event_to_final_stops_drain(self) -> None:
        entered: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            finish = active.to(end)
            stop = idle.to(end) | active.to(end)

            def on_enter_idle(self) -> None:
                self.send("go")
                self.send("finish")
                # This should never execute — machine is final after finish
                self.send("go")

            def on_enter_active(self) -> None:
                entered.append("active")

            def on_enter_end(self) -> None:
                entered.append("end")

        sm = SM()
        sm.send("start")
        assert sm.current_state.id == "end"
        assert entered == ["active", "end"]


# ---------------------------------------------------------------------------
# Force flag and kwargs preservation
# ---------------------------------------------------------------------------


class TestForceAndKwargs:
    def test_queued_event_preserves_force_flag(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            running = State("Running")
            end = State("End", final=True)

            start = starting.to(idle)
            activate = idle.to(running)
            stop = running.to(end) | idle.to(end)

            def on_enter_idle(self) -> None:
                # Force-send activate from any state
                self.send("activate", f=True)

        sm = SM()
        sm.send("start")
        assert sm.current_state.id == "running"

    def test_queued_event_preserves_kwargs(self) -> None:
        received: list[dict] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            stop = idle.to(end) | active.to(end)

            def on_enter_idle(self) -> None:
                self.send("go", data="hello", count=42)

            def on_go(self, **kwargs: object) -> None:
                received.append(dict(kwargs))

        sm = SM()
        sm.send("start")
        assert received == [{"data": "hello", "count": 42}]


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


class TestExceptionSafety:
    def test_exception_in_hook_clears_queue(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            stop = idle.to(end) | active.to(end)

            def on_enter_idle(self) -> None:
                self.send("go")
                raise RuntimeError("boom")

        sm = SM()
        with pytest.raises(RuntimeError, match="boom"):
            sm.send("start")
        # Queue should be cleared, processing flag reset
        assert len(sm._queue) == 0
        assert sm._processing is False

    def test_exception_in_queued_event_clears_queue(self) -> None:
        """Exception during _drain_queue (not initial transition) clears queue."""

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            finish = active.to(end)
            stop = idle.to(end) | active.to(end)

            def on_enter_idle(self) -> None:
                self.send("go")
                self.send("finish")  # should be discarded after exception

            def on_enter_active(self) -> None:
                raise RuntimeError("boom in drain")

        sm = SM()
        with pytest.raises(RuntimeError, match="boom in drain"):
            sm.send("start")
        assert len(sm._queue) == 0
        assert sm._processing is False
        # State is active because assignment happened before on_enter_active raised
        assert sm.current_state.id == "active"

    def test_processing_flag_reset_on_exception(self) -> None:
        """After an exception, subsequent send() calls work normally."""

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)

            start = starting.to(idle)
            stop = idle.to(end)

            def on_enter_idle(self) -> None:
                raise RuntimeError("boom")

        sm = SM()
        with pytest.raises(RuntimeError):
            sm.send("start")
        # Machine should still be functional (state was set before hook)
        assert sm._processing is False
        # State is idle because the assignment happened before on_enter_idle
        assert sm.current_state.id == "idle"
        # Subsequent send should work
        assert sm.send("stop") is True
        assert sm.current_state.id == "end"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_queued_event_with_no_valid_transition_is_discarded(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            active = State("Active")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(active)
            stop = idle.to(end) | active.to(end)

            def on_enter_idle(self) -> None:
                # 'go' transitions idle→active, but then 'go' again has no
                # transition from 'active', so it's discarded
                self.send("go")
                self.send("go")

        sm = SM()
        sm.send("start")
        assert sm.current_state.id == "active"

    def test_no_regression_simple_send(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        assert sm.send("start") is True
        assert sm.current_state.id == "idle"
        assert sm.send("stop") is True
        assert sm.current_state.id == "end"

    def test_no_regression_guard_conditions(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end, cond="is_allowed")

            def is_allowed(self) -> bool:
                return False

        sm = SM()
        sm.send("start")
        assert sm.send("stop") is False
        assert sm.current_state.id == "idle"

    def test_no_regression_internal_transitions(self) -> None:
        log: list[str] = []

        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            tick = idle.to.itself()
            stop = idle.to(end)

            def on_tick(self) -> None:
                log.append("tick")

        sm = SM()
        sm.send("start")
        sm.send("tick")
        sm.send("tick")
        assert log == ["tick", "tick"]
        assert sm.current_state.id == "idle"

    def test_no_regression_event_descriptor(self) -> None:
        class SM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            end = State("End", final=True)
            start = starting.to(idle)
            stop = idle.to(end)

        sm = SM()
        sm.start()
        assert sm.current_state.id == "idle"
        sm.stop()
        assert sm.current_state.id == "end"


# ---------------------------------------------------------------------------
# Integration: clock-style pattern
# ---------------------------------------------------------------------------


class TestClockStyleIntegration:
    """Mirrors the ClockStateMachine pattern end-to-end."""

    def test_idle_activate_deactivate_pattern(self) -> None:
        log: list[str] = []

        class ClockLikeMachine(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            running = State("Running")
            end = State("End", final=True)

            start = starting.to(idle)
            activate = idle.to(running)
            deactivate = running.to(idle)
            tick = running.to.itself()
            stop = running.to(end) | idle.to(end)

            def __init__(self, ready: bool = True) -> None:
                super().__init__()
                self._ready = ready

            def on_enter_idle(self) -> None:
                log.append(f"on_enter_idle:state={self._current_state.id}")
                if self._ready:
                    self.send("activate")

            def on_enter_running(self) -> None:
                log.append(f"on_enter_running:state={self._current_state.id}")
                if not self._ready:
                    self.send("deactivate")

            def after_transition(self, event: str, state: object) -> None:
                log.append(f"after_transition({event}):state={self._current_state.id}")

        # Ready=True: start → idle → (queued activate) → running
        sm = ClockLikeMachine(ready=True)
        sm.send("start")
        assert sm.current_state.id == "running"
        assert log == [
            "on_enter_idle:state=idle",
            "after_transition(start):state=idle",
            "on_enter_running:state=running",
            "after_transition(activate):state=running",
        ]

    def test_idle_activate_deactivate_not_ready(self) -> None:
        log: list[str] = []

        class ClockLikeMachine(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            running = State("Running")
            end = State("End", final=True)

            start = starting.to(idle)
            activate = idle.to(running)
            deactivate = running.to(idle)
            tick = running.to.itself()
            stop = running.to(end) | idle.to(end)

            def __init__(self, ready: bool = True) -> None:
                super().__init__()
                self._ready = ready

            def on_enter_idle(self) -> None:
                log.append(f"on_enter_idle:state={self._current_state.id}")
                if self._ready:
                    self.send("activate")

            def on_enter_running(self) -> None:
                log.append(f"on_enter_running:state={self._current_state.id}")
                if not self._ready:
                    self.send("deactivate")

        # Ready=False: start → idle (stays, no activate queued)
        sm = ClockLikeMachine(ready=False)
        sm.send("start")
        assert sm.current_state.id == "idle"
        assert log == ["on_enter_idle:state=idle"]
