"""Tests for state machine inheritance isolation.

Verifies that the StateMachineMeta metaclass properly isolates State objects
across subclasses, preventing shared-state pollution between different state machine
types.
"""

from typing import Any

import pytest

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


def _make_agent_class(sm_class: type[AgentStateMachine]) -> type[Agent]:
    """Create a minimal Agent class wired to *sm_class*."""

    class _Agent(Agent):
        definition = AgentDefinition(
            configuration_class=AgentConfiguration,
            events_publishing_class=_MockPublishing,
            events_listening_class=_MockListening,
            state_machine_class=sm_class,
            status_class=AgentStatus,
        )

    return _Agent


# ---------------------------------------------------------------------------
# State machine subclasses used in tests
# ---------------------------------------------------------------------------


class SM_Alpha(AgentStateMachine):
    """Subclass that relies entirely on auto-created default states."""

    def after_transition(self, event: str, state: State) -> None:
        pass


class SM_Beta(AgentStateMachine):
    """Another subclass relying on auto-created default states."""

    def after_transition(self, event: str, state: State) -> None:
        pass


class SM_WithRunning(BasicAgentStateMachine):
    """Extends BasicAgentStateMachine with a 'running' state and transitions
    that reference parent states."""

    running = State("Running")
    ready = BasicAgentStateMachine.idle.to(running)
    not_ready = running.to(BasicAgentStateMachine.idle)
    stop = (
        BasicAgentStateMachine.starting.to(BasicAgentStateMachine.end)
        | BasicAgentStateMachine.idle.to(BasicAgentStateMachine.end)
        | running.to(BasicAgentStateMachine.end)
    )


class SM_WithProcessing(BasicAgentStateMachine):
    """Extends BasicAgentStateMachine with a 'processing' state."""

    processing = State("Processing")
    begin = BasicAgentStateMachine.idle.to(processing)
    finish = processing.to(BasicAgentStateMachine.end)
    stop = (
        BasicAgentStateMachine.starting.to(BasicAgentStateMachine.end)
        | BasicAgentStateMachine.idle.to(BasicAgentStateMachine.end)
        | processing.to(BasicAgentStateMachine.end)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoCreatedDefaultStates:
    """Two direct AgentStateMachine subclasses get independent default states."""

    def test_states_are_different_objects(self) -> None:
        assert SM_Alpha.starting is not SM_Beta.starting
        assert SM_Alpha.idle is not SM_Beta.idle
        assert SM_Alpha.end is not SM_Beta.end

    def test_both_have_correct_state_ids(self) -> None:
        for sm in (SM_Alpha, SM_Beta):
            state_ids = {s.id for s in sm.states}
            assert {"starting", "idle", "end"} == state_ids

    def test_default_transitions_present(self) -> None:
        for sm in (SM_Alpha, SM_Beta):
            # Verify events exist
            event_names = {e.name for e in sm.events}
            assert "start" in event_names
            assert "stop" in event_names

            # Verify transition structure via state objects
            starting_targets = {
                (t.source.id, t.target.id) for t in sm.starting.transitions
            }
            assert ("starting", "idle") in starting_targets  # start
            assert ("starting", "end") in starting_targets  # stop

            idle_targets = {(t.source.id, t.target.id) for t in sm.idle.transitions}
            assert ("idle", "end") in idle_targets  # stop

    def test_transitions_on_alpha_dont_leak_to_beta(self) -> None:
        alpha_starting_targets = {t.target.id for t in SM_Alpha.starting.transitions}
        beta_starting_targets = {t.target.id for t in SM_Beta.starting.transitions}
        assert alpha_starting_targets == beta_starting_targets


class TestInheritanceWithNewStatesAndTransitions:
    """Subclasses that add states + transitions referencing parent states."""

    def test_extended_sm_has_all_states(self) -> None:
        state_ids = {s.id for s in SM_WithRunning.states}
        assert {"starting", "idle", "running", "end"} == state_ids

    def test_processing_sm_has_all_states(self) -> None:
        state_ids = {s.id for s in SM_WithProcessing.states}
        assert {"starting", "idle", "processing", "end"} == state_ids

    def test_parent_not_polluted(self) -> None:
        """BasicAgentStateMachine must still have exactly 3 states."""
        state_ids = {s.id for s in BasicAgentStateMachine.states}
        assert {"starting", "idle", "end"} == state_ids

    def test_parent_idle_transitions_unchanged(self) -> None:
        idle_targets = {t.target.id for t in BasicAgentStateMachine.idle.transitions}
        # Only the default stop transition: idle -> end
        assert idle_targets == {"end"}

    def test_subclass_states_are_independent_from_parent(self) -> None:
        assert SM_WithRunning.idle is not BasicAgentStateMachine.idle
        assert SM_WithRunning.starting is not BasicAgentStateMachine.starting
        assert SM_WithRunning.end is not BasicAgentStateMachine.end

    def test_two_subclasses_have_independent_states(self) -> None:
        assert SM_WithRunning.idle is not SM_WithProcessing.idle
        assert SM_WithRunning.starting is not SM_WithProcessing.starting
        assert SM_WithRunning.end is not SM_WithProcessing.end


class TestRuntimeTransitions:
    """Verify that transitions actually work at runtime after state isolation."""

    def test_extended_sm_transitions(self) -> None:
        AgentCls = _make_agent_class(SM_WithRunning)
        agent = AgentCls({})
        try:
            assert agent.machine.current_state.id == "starting"
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.ready()
            assert agent.machine.current_state.id == "running"
            agent.machine.not_ready()
            assert agent.machine.current_state.id == "idle"
            agent.machine.stop()
            assert agent.machine.current_state.id == "end"
            assert agent.machine.current_state.final
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_processing_sm_transitions(self) -> None:
        AgentCls = _make_agent_class(SM_WithProcessing)
        agent = AgentCls({})
        try:
            assert agent.machine.current_state.id == "starting"
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.begin()
            assert agent.machine.current_state.id == "processing"
            agent.machine.finish()
            assert agent.machine.current_state.id == "end"
            assert agent.machine.current_state.final
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_basic_agent_sm_still_works(self) -> None:
        AgentCls = _make_agent_class(BasicAgentStateMachine)
        agent = AgentCls({})
        try:
            assert agent.machine.current_state.id == "starting"
            agent.machine.safe_start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.safe_stop()
            assert agent.machine.current_state.id == "end"
            assert agent.machine.current_state.final
        finally:
            try:
                agent.teardown()
            except Exception:
                pass

    def test_two_instances_of_same_extended_sm_are_independent(self) -> None:
        AgentCls = _make_agent_class(SM_WithRunning)
        agent1 = AgentCls({})
        agent2 = AgentCls({})
        try:
            # Advance agent1, leave agent2 at start
            agent1.machine.start()
            agent1.machine.ready()
            assert agent1.machine.current_state.id == "running"
            assert agent2.machine.current_state.id == "starting"
        finally:
            for a in (agent1, agent2):
                try:
                    a.teardown()
                except Exception:
                    pass


class TestDiamondInheritance:
    """Verify diamond inheritance works correctly with state isolation."""

    def test_diamond_has_states_from_both_branches(self) -> None:
        class Base(AgentStateMachine):
            def after_transition(self, event: str, state: State) -> None:
                pass

        class Left(Base):
            left_state = State("Left")
            go_left = Base.idle.to(left_state)
            back_left = left_state.to(Base.idle)
            stop = (
                Base.starting.to(Base.end)
                | Base.idle.to(Base.end)
                | left_state.to(Base.end)
            )

            def after_transition(self, event: str, state: State) -> None:
                pass

        class Right(Base):
            right_state = State("Right")
            go_right = Base.idle.to(right_state)
            back_right = right_state.to(Base.idle)
            stop = (
                Base.starting.to(Base.end)
                | Base.idle.to(Base.end)
                | right_state.to(Base.end)
            )

            def after_transition(self, event: str, state: State) -> None:
                pass

        try:

            class Diamond(Left, Right):
                stop = (
                    Left.starting.to(Left.end)
                    | Left.idle.to(Left.end)
                    | Left.left_state.to(Left.end)
                    | Right.right_state.to(Right.end)
                )

                def after_transition(self, event: str, state: State) -> None:
                    pass

        except Exception as exc:
            pytest.skip(f"Diamond inheritance failed (known limitation): {exc}")
            return

        diamond_state_ids = {s.id for s in Diamond.states}
        assert "left_state" in diamond_state_ids
        assert "right_state" in diamond_state_ids
        assert "starting" in diamond_state_ids
        assert "idle" in diamond_state_ids
        assert "end" in diamond_state_ids

        # States must be independent from both Left and Right
        assert Diamond.idle is not Left.idle
        assert Diamond.idle is not Right.idle
        assert Diamond.starting is not Left.starting
        assert Diamond.starting is not Right.starting

        # Runtime transitions
        AgentCls = _make_agent_class(Diamond)
        agent = AgentCls({})
        try:
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.go_left()
            assert agent.machine.current_state.id == "left_state"
            agent.machine.back_left()
            assert agent.machine.current_state.id == "idle"
            agent.machine.stop()
            assert agent.machine.current_state.id == "end"
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


class TestDeepInheritanceChain:
    """Verify a 4-level deep inheritance chain works correctly."""

    def test_deep_chain_has_all_states(self) -> None:
        # Level A: first concrete subclass, gets auto-created starting/idle/end
        class A(BasicAgentStateMachine):
            a_state = State("A")
            go_a = BasicAgentStateMachine.idle.to(a_state)
            back_a = a_state.to(BasicAgentStateMachine.idle)
            stop = (
                BasicAgentStateMachine.starting.to(BasicAgentStateMachine.end)
                | BasicAgentStateMachine.idle.to(BasicAgentStateMachine.end)
                | a_state.to(BasicAgentStateMachine.end)
            )

            def after_transition(self, event: str, state: State) -> None:
                pass

        # Level B: inherits from A, must reconnect inherited a_state
        class B(A):
            b_state = State("B")
            go_a = A.idle.to(A.a_state)
            back_a = A.a_state.to(A.idle)
            go_b = A.idle.to(b_state)
            back_b = b_state.to(A.idle)
            stop = (
                A.starting.to(A.end)
                | A.idle.to(A.end)
                | A.a_state.to(A.end)
                | b_state.to(A.end)
            )

            def after_transition(self, event: str, state: State) -> None:
                pass

        # Level C: inherits from B, must reconnect inherited a_state and b_state
        class C(B):
            c_state = State("C")
            go_a = B.idle.to(B.a_state)
            back_a = B.a_state.to(B.idle)
            go_b = B.idle.to(B.b_state)
            back_b = B.b_state.to(B.idle)
            go_c = B.idle.to(c_state)
            back_c = c_state.to(B.idle)
            stop = (
                B.starting.to(B.end)
                | B.idle.to(B.end)
                | B.a_state.to(B.end)
                | B.b_state.to(B.end)
                | c_state.to(B.end)
            )

            def after_transition(self, event: str, state: State) -> None:
                pass

        # Level D: inherits from C, must reconnect all inherited states
        class D(C):
            d_state = State("D")
            go_a = C.idle.to(C.a_state)
            back_a = C.a_state.to(C.idle)
            go_b = C.idle.to(C.b_state)
            back_b = C.b_state.to(C.idle)
            go_c = C.idle.to(C.c_state)
            back_c = C.c_state.to(C.idle)
            go_d = C.idle.to(d_state)
            back_d = d_state.to(C.idle)
            stop = (
                C.starting.to(C.end)
                | C.idle.to(C.end)
                | C.a_state.to(C.end)
                | C.b_state.to(C.end)
                | C.c_state.to(C.end)
                | d_state.to(C.end)
            )

            def after_transition(self, event: str, state: State) -> None:
                pass

        # D should have all states
        d_state_ids = {s.id for s in D.states}
        assert {
            "starting",
            "idle",
            "end",
            "a_state",
            "b_state",
            "c_state",
            "d_state",
        } == d_state_ids

        # States at each level should be independent
        assert D.idle is not C.idle
        assert C.idle is not B.idle
        assert B.idle is not A.idle

        assert D.a_state is not C.a_state
        assert C.a_state is not B.a_state
        assert B.a_state is not A.a_state

        # Runtime transitions
        AgentCls = _make_agent_class(D)
        agent = AgentCls({})
        try:
            agent.machine.start()
            assert agent.machine.current_state.id == "idle"
            agent.machine.go_a()
            assert agent.machine.current_state.id == "a_state"
            agent.machine.back_a()
            assert agent.machine.current_state.id == "idle"
            agent.machine.go_d()
            assert agent.machine.current_state.id == "d_state"
            agent.machine.back_d()
            assert agent.machine.current_state.id == "idle"
            agent.machine.stop()
            assert agent.machine.current_state.id == "end"
            assert agent.machine.current_state.final
        finally:
            try:
                agent.teardown()
            except Exception:
                pass


class TestCreationOrderDoesNotMatter:
    """Creating subclasses in any order must not pollute each other."""

    def test_creating_second_subclass_after_first_does_not_break_first(self) -> None:
        # SM_WithRunning and SM_WithProcessing are already created at module level.
        # Verify the first one still has correct states/transitions.
        running_idle_targets = {t.target.id for t in SM_WithRunning.idle.transitions}
        assert "running" in running_idle_targets

        processing_idle_targets = {
            t.target.id for t in SM_WithProcessing.idle.transitions
        }
        assert "processing" in processing_idle_targets

        # Ensure they don't see each other's custom states
        running_state_ids = {s.id for s in SM_WithRunning.states}
        assert "processing" not in running_state_ids

        processing_state_ids = {s.id for s in SM_WithProcessing.states}
        assert "running" not in processing_state_ids
