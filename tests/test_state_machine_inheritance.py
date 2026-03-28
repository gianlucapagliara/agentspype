"""Tests for state machine inheritance: transition remapping, merging, and callbacks.

These tests exercise the metaclass behaviours that enable real-world inheritance
patterns found in downstream projects (e.g. bl_agents):

* Inherited transitions are automatically remapped to fresh state clones.
* When a subclass redefines an event (e.g. ``stop``), the parent's transitions
  for that event are **merged** with the subclass's, not silently dropped.
* Transition metadata (``cond``, ``unless``, ``before``, ``on``, ``after``) is
  preserved through remapping.
* Convention callbacks (``on_stop``, ``on_start``, …) fire correctly on both
  explicitly-defined and inherited transitions.
"""

from typing import Any

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine, BasicAgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.fsm import State

# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------


class _MockListening(AgentListening):
    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass


class _MockPublishing(StateAgentPublishing):
    def publish(self, event_publication: Any, event_data: Any) -> None:
        pass


def _make_agent(sm_class: type[AgentStateMachine]) -> Agent:
    """Create a minimal Agent instance wired to *sm_class*."""

    class _Agent(Agent):
        definition = AgentDefinition(
            configuration_class=AgentConfiguration,
            events_publishing_class=_MockPublishing,
            events_listening_class=_MockListening,
            state_machine_class=sm_class,
            status_class=AgentStatus,
        )

    return _Agent({})


# ---------------------------------------------------------------------------
# Hierarchy used across multiple test classes
#
#   BasicAgentStateMachine          (starting, idle, end)
#         |
#     BaseSM                        + running, ready, not_ready
#         |
#     ChildSM                       + waiting, launch, finish
#                                     overrides stop (partial)
# ---------------------------------------------------------------------------


class BaseSM(BasicAgentStateMachine):
    """Adds a 'running' state with ready/not_ready transitions.

    Resembles ``CompleteHorologistStateMachine`` in bl_agents.
    """

    running = State("Running")

    ready = BasicAgentStateMachine.idle.to(running)
    not_ready = running.to(BasicAgentStateMachine.idle)
    # Extends stop to also cover running → end
    stop = running.to(BasicAgentStateMachine.end)


class ChildSM(BaseSM):
    """Adds a 'waiting' state and partially overrides stop.

    Resembles ``StandardOrderAgentStateMachine`` in bl_agents: the subclass
    defines ``stop = waiting.to.itself(internal=True)`` which must be **merged**
    with the parent's ``stop`` transitions, not replace them.
    """

    waiting = State("Waiting")

    launch = BaseSM.running.to(waiting)
    finish = waiting.to(BaseSM.end)

    # Partial override: only defines stop behaviour from 'waiting'.
    # Parent's stop transitions (running→end, starting→end, idle→end) must
    # be merged in automatically.
    stop = waiting.to.itself(internal=True)

    tick = BaseSM.idle.to.itself() | BaseSM.running.to.itself()


# ---------------------------------------------------------------------------
# Tests: inherited transition auto-remapping
# ---------------------------------------------------------------------------


class TestInheritedTransitionRemapping:
    """Subclass that does NOT redefine all parent transitions still gets
    a fully-connected graph with remapped states."""

    def test_child_has_all_states(self) -> None:
        state_ids = {s.id for s in ChildSM.states}
        assert state_ids == {"starting", "idle", "running", "end", "waiting"}

    def test_child_states_are_independent_from_parent(self) -> None:
        for attr in ("starting", "idle", "end", "running"):
            assert getattr(ChildSM, attr) is not getattr(BaseSM, attr)

    def test_parent_not_polluted_by_child(self) -> None:
        """BaseSM must not gain 'waiting' or any child-only transitions."""
        state_ids = {s.id for s in BaseSM.states}
        assert "waiting" not in state_ids

    def test_inherited_ready_transition_present(self) -> None:
        """'ready' (idle→running) is inherited from BaseSM, not redefined
        in ChildSM.  It must be automatically remapped."""
        idle_targets = {t.target.id for t in ChildSM.idle.transitions}
        assert "running" in idle_targets

    def test_inherited_not_ready_transition_present(self) -> None:
        """'not_ready' (running→idle) is inherited from BaseSM."""
        running_targets = {t.target.id for t in ChildSM.running.transitions}
        assert "idle" in running_targets

    def test_inherited_start_transition_present(self) -> None:
        """'start' (starting→idle) is inherited from the default transitions."""
        starting_targets = {t.target.id for t in ChildSM.starting.transitions}
        assert "idle" in starting_targets


# ---------------------------------------------------------------------------
# Tests: event-name merging (stop override)
# ---------------------------------------------------------------------------


class TestEventNameMerging:
    """When a subclass redefines an event, the parent's transitions for that
    event must be merged, not replaced."""

    def test_stop_from_waiting_is_internal(self) -> None:
        """The subclass-defined stop (waiting→waiting, internal) must exist."""
        waiting_stop_transitions = [
            t for t in ChildSM.waiting.transitions if t.event == "stop" and t.internal
        ]
        assert len(waiting_stop_transitions) >= 1

    def test_stop_from_running_reaches_end(self) -> None:
        """The parent's stop (running→end) must be merged in."""
        running_stop_to_end = [
            t
            for t in ChildSM.running.transitions
            if t.event == "stop" and t.target.id == "end"
        ]
        assert len(running_stop_to_end) >= 1

    def test_stop_from_idle_reaches_end(self) -> None:
        """Default stop (idle→end) must be merged in."""
        idle_stop_to_end = [
            t
            for t in ChildSM.idle.transitions
            if t.event == "stop" and t.target.id == "end"
        ]
        assert len(idle_stop_to_end) >= 1

    def test_stop_from_starting_reaches_end(self) -> None:
        """Default stop (starting→end) must be merged in."""
        starting_stop_to_end = [
            t
            for t in ChildSM.starting.transitions
            if t.event == "stop" and t.target.id == "end"
        ]
        assert len(starting_stop_to_end) >= 1


# ---------------------------------------------------------------------------
# Tests: condition and callback preservation through remapping
# ---------------------------------------------------------------------------


class ChildWithConditions(BaseSM):
    """Subclass that uses ``cond`` and ``unless`` on transitions referencing
    parent states.  These must survive remapping."""

    waiting = State("Waiting")

    launch = BaseSM.running.to(waiting)
    finish = waiting.to(BaseSM.end, cond="all_done") | waiting.to.itself(
        internal=True, unless="all_done"
    )
    stop = waiting.to.itself(internal=True)
    tick = BaseSM.idle.to.itself() | BaseSM.running.to.itself()

    def all_done(self) -> bool:
        return getattr(self, "_all_done", False)


class TestConditionPreservation:
    """Transition conditions must survive remapping through the metaclass."""

    def test_finish_stays_in_waiting_when_condition_false(self) -> None:
        agent = _make_agent(ChildWithConditions)
        try:
            agent.machine.start()
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"

            # Condition is False → should stay in waiting
            agent.machine._all_done = False
            agent.machine.finish()
            assert agent.machine.current_state.id == "waiting"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_finish_goes_to_end_when_condition_true(self) -> None:
        agent = _make_agent(ChildWithConditions)
        try:
            agent.machine.start()
            agent.machine.ready()
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"

            # Condition is True → should go to end
            agent.machine._all_done = True
            agent.machine.finish()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests: convention callbacks fire on inherited transitions
# ---------------------------------------------------------------------------


class TestConventionCallbacks:
    """Convention callbacks (on_stop, on_start, etc.) must fire correctly on
    both explicitly-defined and inherited transitions."""

    def test_on_stop_fires_from_explicit_transition(self) -> None:
        """on_stop fires when stop is called from 'waiting' (subclass-defined)."""
        agent = _make_agent(ChildSM)
        try:
            agent.machine.start()
            agent.machine.ready()
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"
            agent.machine.stop()
            assert agent.machine.should_stop()
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_on_stop_fires_from_inherited_transition(self) -> None:
        """on_stop fires when stop is called from 'idle' (inherited default)."""
        agent = _make_agent(ChildSM)
        try:
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.stop()
            assert agent.machine.should_stop()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_on_stop_fires_from_running(self) -> None:
        """on_stop fires when stop is called from 'running' (parent-defined, merged)."""
        agent = _make_agent(ChildSM)
        try:
            agent.machine.start()
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.stop()
            assert agent.machine.should_stop()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_on_start_fires(self) -> None:
        """on_start fires on the inherited 'start' transition."""
        agent = _make_agent(ChildSM)
        try:
            # on_start calls agent.listening.subscribe()
            # which is _MockListening.subscribe() — just check no error
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests: runtime full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end runtime tests through complete state machine lifecycles."""

    def test_child_full_lifecycle(self) -> None:
        agent = _make_agent(ChildSM)
        try:
            assert agent.machine.current_state.id == "starting"
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.not_ready()
            assert agent.machine.current_state.id == "idle"
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"
            agent.machine.finish()
            assert agent.machine.current_state.id == "end"
            assert agent.machine.current_state.final
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_child_stop_from_every_non_final_state(self) -> None:
        """stop must be reachable from starting, idle, running, and waiting."""
        for target_state, steps in [
            ("starting", []),
            ("idle", ["start"]),
            ("running", ["start", "ready"]),
            ("waiting", ["start", "ready", "launch"]),
        ]:
            agent = _make_agent(ChildSM)
            try:
                for step in steps:
                    getattr(agent.machine, step)()
                assert agent.machine.current_state.id == target_state
                agent.machine.stop()
                assert agent.machine.should_stop()
            finally:
                try:
                    agent.teardown()
                except Exception:
                    pass

    def test_tick_self_loops_dont_change_state(self) -> None:
        """tick (self-loop) keeps the machine in the same state."""
        agent = _make_agent(ChildSM)
        try:
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.tick()
            assert agent.machine.current_state.id == "idle"

            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.tick()
            assert agent.machine.current_state.id == "running"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests: multiple inheritance with mixin pattern
# ---------------------------------------------------------------------------


class MixinSM(AgentStateMachine):
    """Abstract mixin that declares a 'running' state annotation and
    readiness-checking logic.  Resembles ``HorologistStateMachine``."""

    running: State  # annotation only, no default

    def after_transition(self, event: str, state: State) -> None:
        pass


class ConcreteMixinSM(MixinSM, BasicAgentStateMachine):
    """Concrete class combining the mixin with BasicAgentStateMachine.
    Resembles ``CompleteHorologistStateMachine``."""

    running = State("Running")

    # Must reference MixinSM (first base) for state resolution
    ready = MixinSM.idle.to(running)
    not_ready = running.to(MixinSM.idle)
    stop = running.to(MixinSM.end)


class SubOfMixin(ConcreteMixinSM):
    """Subclass of the mixin-based concrete class.
    Resembles ``StandardOrderAgentStateMachine``."""

    waiting = State("Waiting")
    launch = ConcreteMixinSM.running.to(waiting)
    finish = waiting.to(ConcreteMixinSM.end)
    stop = waiting.to.itself(internal=True)
    tick = ConcreteMixinSM.idle.to.itself() | ConcreteMixinSM.running.to.itself()


class TestMultipleInheritanceMixin:
    """Multiple inheritance with an abstract mixin + concrete base."""

    def test_concrete_mixin_has_all_states(self) -> None:
        state_ids = {s.id for s in ConcreteMixinSM.states}
        assert state_ids == {"starting", "idle", "running", "end"}

    def test_sub_of_mixin_has_all_states(self) -> None:
        state_ids = {s.id for s in SubOfMixin.states}
        assert state_ids == {"starting", "idle", "running", "end", "waiting"}

    def test_sub_of_mixin_states_independent(self) -> None:
        for attr in ("starting", "idle", "end", "running"):
            assert getattr(SubOfMixin, attr) is not getattr(ConcreteMixinSM, attr)

    def test_inherited_ready_present(self) -> None:
        idle_targets = {t.target.id for t in SubOfMixin.idle.transitions}
        assert "running" in idle_targets

    def test_stop_merged(self) -> None:
        """stop must include waiting→waiting AND running→end (from parent)."""
        running_stop_to_end = [
            t
            for t in SubOfMixin.running.transitions
            if t.event == "stop" and t.target.id == "end"
        ]
        assert len(running_stop_to_end) >= 1

    def test_full_lifecycle(self) -> None:
        agent = _make_agent(SubOfMixin)
        try:
            agent.machine.start()
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"
            agent.machine.finish()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests: intermediate class without stop (variant pattern)
# ---------------------------------------------------------------------------


class ConcreteMixinNoStop(MixinSM, BasicAgentStateMachine):
    """Like ConcreteMixinSM but with a minimal stop for running.
    Resembles ``CompleteHorologistStateMachineWithoutStop``."""

    running = State("Running")

    ready = MixinSM.idle.to(running)
    not_ready = running.to(MixinSM.idle)
    stop = running.to(MixinSM.end)


class SubOfNoStop(ConcreteMixinNoStop):
    """Subclass that defines stop, launch, finish.
    Resembles ``TriggerExecutorAgentStateMachine``."""

    waiting = State("Waiting")

    launch = ConcreteMixinNoStop.running.to(
        waiting
    ) | ConcreteMixinNoStop.idle.to.itself(internal=True)
    finish = waiting.to(ConcreteMixinNoStop.end)
    stop = waiting.to.itself(internal=True) | ConcreteMixinNoStop.running.to(waiting)
    tick = (
        ConcreteMixinNoStop.idle.to.itself() | ConcreteMixinNoStop.running.to.itself()
    )


class TestNoStopVariant:
    """Intermediate class without explicit stop from running."""

    def test_has_all_states(self) -> None:
        state_ids = {s.id for s in SubOfNoStop.states}
        assert state_ids == {"starting", "idle", "running", "end", "waiting"}

    def test_stop_includes_default_paths(self) -> None:
        """Default idle→end and starting→end should be merged in."""
        idle_stop = [
            t
            for t in SubOfNoStop.idle.transitions
            if t.event == "stop" and t.target.id == "end"
        ]
        assert len(idle_stop) >= 1

    def test_full_lifecycle(self) -> None:
        agent = _make_agent(SubOfNoStop)
        try:
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"
            agent.machine.finish()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_stop_from_running_goes_to_waiting(self) -> None:
        """SubOfNoStop defines stop with running→waiting."""
        agent = _make_agent(SubOfNoStop)
        try:
            agent.machine.start()
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.stop()
            # running → waiting (from subclass stop definition)
            assert agent.machine.current_state.id == "waiting"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests: _transition_lists_ storage
# ---------------------------------------------------------------------------


class TestTransitionListsStorage:
    """The ``_transition_lists_`` class attribute must be populated correctly."""

    def test_basic_sm_has_transition_lists(self) -> None:
        tls = BasicAgentStateMachine._transition_lists_
        assert "start" in tls
        assert "stop" in tls

    def test_child_has_all_transition_lists(self) -> None:
        tls = ChildSM._transition_lists_
        # Subclass-defined
        assert "launch" in tls
        assert "finish" in tls
        assert "tick" in tls
        # Merged (subclass + parent)
        assert "stop" in tls
        # Inherited (remapped from parent)
        assert "ready" in tls
        assert "not_ready" in tls
        assert "start" in tls

    def test_transition_lists_values_are_transition_lists(self) -> None:
        from agentspype.fsm import TransitionList

        for name, tl in ChildSM._transition_lists_.items():
            assert isinstance(tl, TransitionList), (
                f"_transition_lists_['{name}'] is {type(tl)}, expected TransitionList"
            )


# ---------------------------------------------------------------------------
# Tests: grandchild inherits merged transitions
# ---------------------------------------------------------------------------


class GrandchildSM(ChildSM):
    """Third level: inherits from ChildSM which already has merged stop."""

    processing = State("Processing")
    process = ChildSM.waiting.to(processing)
    done = processing.to(ChildSM.end)
    stop = processing.to(ChildSM.end)


class TestGrandchildInheritance:
    """Three-level inheritance: grandchild gets all merged transitions."""

    def test_has_all_states(self) -> None:
        state_ids = {s.id for s in GrandchildSM.states}
        assert state_ids == {
            "starting",
            "idle",
            "running",
            "end",
            "waiting",
            "processing",
        }

    def test_states_independent_from_child(self) -> None:
        for attr in ("starting", "idle", "end", "running", "waiting"):
            assert getattr(GrandchildSM, attr) is not getattr(ChildSM, attr)

    def test_stop_still_merged(self) -> None:
        """stop must still reach end from idle, running, starting."""
        for state_name in ("idle", "running", "starting"):
            state = getattr(GrandchildSM, state_name)
            stop_to_end = [
                t
                for t in state.transitions
                if t.event == "stop" and t.target.id == "end"
            ]
            assert len(stop_to_end) >= 1, f"No stop→end transition from {state_name}"

    def test_full_lifecycle(self) -> None:
        agent = _make_agent(GrandchildSM)
        try:
            agent.machine.start()
            agent.machine.ready()
            agent.machine.launch()
            assert agent.machine.current_state.id == "waiting"
            agent.machine.process()
            assert agent.machine.current_state.id == "processing"
            agent.machine.done()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_stop_from_every_state(self) -> None:
        """stop must work from every non-final state."""
        for target_state, steps in [
            ("starting", []),
            ("idle", ["start"]),
            ("running", ["start", "ready"]),
            ("waiting", ["start", "ready", "launch"]),
            ("processing", ["start", "ready", "launch", "process"]),
        ]:
            agent = _make_agent(GrandchildSM)
            try:
                for step in steps:
                    getattr(agent.machine, step)()
                assert agent.machine.current_state.id == target_state
                agent.machine.stop()
                assert agent.machine.should_stop()
            finally:
                try:
                    agent.teardown()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Tests: stop transition validation
# ---------------------------------------------------------------------------


class TestStopTransitionValidation:
    """AgentStateMachine metaclass must reject classes with non-final states
    missing a 'stop' transition."""

    def test_missing_stop_raises_error(self) -> None:
        """A custom state without stop must cause ValueError at class creation."""
        import pytest

        with pytest.raises(ValueError, match="Missing 'stop' for: running"):

            class BadSM(AgentStateMachine):
                running = State("Running")
                go = AgentStateMachine.idle.to(running)

                def after_transition(self, event: str, state: State) -> None:
                    pass

    def test_multiple_missing_states_listed(self) -> None:
        """Error message lists all states missing stop."""
        import pytest

        with pytest.raises(ValueError, match="processing") as exc_info:

            class BadSM2(AgentStateMachine):
                running = State("Running")
                processing = State("Processing")
                go = AgentStateMachine.idle.to(running)
                process = running.to(processing)

                def after_transition(self, event: str, state: State) -> None:
                    pass

        assert "running" in str(exc_info.value)
        assert "processing" in str(exc_info.value)

    def test_internal_stop_satisfies_validation(self) -> None:
        """An internal self-transition on stop counts as defining stop."""

        class InternalStopSM(AgentStateMachine):
            running = State("Running")
            go = AgentStateMachine.idle.to(running)
            stop = running.to.itself(internal=True)

            def after_transition(self, event: str, state: State) -> None:
                pass

        assert "running" in {s.id for s in InternalStopSM.states}

    def test_stop_to_cleanup_state_satisfies_validation(self) -> None:
        """Stop transitioning to a non-end cleanup state is valid."""

        class CleanupSM(AgentStateMachine):
            running = State("Running")
            cleaning = State("Cleaning")
            go = AgentStateMachine.idle.to(running)
            stop = running.to(cleaning) | cleaning.to(AgentStateMachine.end)

            def after_transition(self, event: str, state: State) -> None:
                pass

        assert "cleaning" in {s.id for s in CleanupSM.states}

    def test_default_states_pass_validation(self) -> None:
        """BasicAgentStateMachine with only default states passes."""

        class MinimalSM(AgentStateMachine):
            def after_transition(self, event: str, state: State) -> None:
                pass

        assert "starting" in {s.id for s in MinimalSM.states}

    def test_child_adding_state_must_define_stop(self) -> None:
        """A child class adding a new state must cover it with stop."""
        import pytest

        class ParentSM(AgentStateMachine):
            def after_transition(self, event: str, state: State) -> None:
                pass

        with pytest.raises(ValueError, match="extra"):

            class ChildWithoutStop(ParentSM):
                extra = State("Extra")
                go = ParentSM.idle.to(extra)

                def after_transition(self, event: str, state: State) -> None:
                    pass

    def test_plain_state_machine_not_affected(self) -> None:
        """Plain StateMachine does not enforce stop validation."""
        from agentspype.fsm.machine import StateMachine

        class PlainSM(StateMachine):
            starting = State("Starting", initial=True)
            idle = State("Idle")
            running = State("Running")
            end = State("End", final=True)

            start = starting.to(idle)
            go = idle.to(running)

        assert "running" in {s.id for s in PlainSM.states}
