# State Machines

AgentsPype uses a custom lightweight FSM engine (`agentspype.fsm`). `AgentStateMachine` extends `StateMachine` with agent-specific hooks and a metaclass that injects default states and transitions into every subclass.

## Default Structure

Every `AgentStateMachine` subclass receives the following states and transitions automatically (unless the subclass defines them itself):

| Name | Type | Description |
|---|---|---|
| `starting` | `State(initial=True)` | Initial state; the machine begins here |
| `idle` | `State` | Ready state; the machine waits for work here |
| `end` | `State(final=True)` | Terminal state; triggers agent teardown |
| `start` | Transition | `starting → idle` |
| `stop` | Transition | `starting → end` or `idle → end` |

The `on_enter_end` hook on `AgentStateMachine` calls `agent.teardown()` automatically when the `end` state is entered.

## Declaring States and Transitions

States and transitions are class attributes. You can declare all of them yourself, mix in the defaults, or rely entirely on the defaults.

```python
from agentspype.fsm import State
from agentspype.agent.state_machine import AgentStateMachine


class WorkerStateMachine(AgentStateMachine):
    # States — override the defaults entirely
    starting = State("Starting", initial=True)
    idle = State("Idle")
    processing = State("Processing")
    paused = State("Paused")
    end = State("End", final=True)

    # Transitions
    start = starting.to(idle)
    pick_job = idle.to(processing)
    pause = processing.to(paused)
    resume = paused.to(processing)
    finish = processing.to(idle)
    stop = (
        starting.to(end)
        | idle.to(end)
        | processing.to(end)
        | paused.to(end)
    )

    def after_transition(self, event, state):
        self.agent.publishing.publish_transition(event, state)
```

!!! note
    If you define `starting`, `idle`, or `end` yourself, the metaclass will not overwrite them. If you omit any of them, the metaclass fills in sensible defaults.

## The `AgentStateMachineMeta` Metaclass

`StateMachineMeta` injects default `starting`, `idle`, `end` states and `start`, `stop` transitions into any `StateMachine` subclass that does not define them. It also clones inherited states to ensure full isolation between parent and child classes.

This happens at **class creation time**, before any instance is created. The base class `AgentStateMachine` itself is excluded from this injection.

```python
# This subclass only defines after_transition — everything else is injected
class MinimalMachine(AgentStateMachine):
    def after_transition(self, event, state):
        pass

# MinimalMachine has: starting, idle, end, start, stop
print([s.id for s in MinimalMachine.states])
# ['Starting', 'Idle', 'End']
```

## Required Hook: `after_transition`

`after_transition` is declared `@abstractmethod` in `AgentStateMachine`. Every concrete subclass **must** implement it. It is called after every successful transition.

The most common implementation publishes the transition event:

```python
def after_transition(self, event, state):
    if isinstance(self.agent.publishing, StateAgentPublishing):
        self.agent.publishing.publish_transition(event, state)
```

You can also use it to update status, trigger side effects, or log transitions.

## Built-in Transition Hooks

`AgentStateMachine` provides several hooks that are already implemented:

### `before_transition(event, state, source, target)`

Logs a debug message for every transition where `source != target`. Override to add pre-transition validation or guards.

### `on_start()`

Called when the `start` transition fires. Automatically calls `self.agent.listening.subscribe()`. If you override this, call `super().on_start()` to retain the subscription behavior.

### `on_stop()`

Called when the `stop` transition fires. Sets the internal `_should_stop` flag to `True`.

### `on_enter_end()`

Called when entering the `end` state. Calls `self.agent.teardown()`.

## Supported Convention Hooks

The FSM resolves hook methods by naming convention:

- **`on_enter_<state_id>()`** — called when entering a state
- **`on_exit_<state_id>()`** — called when exiting a state
- **`on_<event_name>(**kwargs)`** — called when an event fires (receives kwargs passed to `send()`)
- **`before_transition(event, state, source, target)`** — global hook called before every transition
- **`after_transition(event, state)`** — global hook called after every transition

The full hook execution order for each transition is:

1. `before_transition(event, state, source, target)` — global
2. `before_<event>(**kwargs)` — per-event
3. `on_<event>(**kwargs)` — per-event
4. `on_exit_<state_id>()` — if not internal transition
5. *[state change]* — if not internal transition
6. `on_enter_<state_id>()` — if not internal transition
7. `after_<event>(**kwargs)` — per-event
8. `after_transition(event, state)` — global

## Safe Start and Stop

Two convenience methods guard against invalid transitions:

```python
# Only calls start() if currently in the initial state
agent.machine.safe_start()

# Only calls stop() if not already in a final state
agent.machine.safe_stop()
```

Both methods use the `f=True` flag internally, which forces the transition even if conditions would otherwise block it.

## Accessing the Agent

`AgentStateMachine` holds a `weakref` to its agent to avoid reference cycles. Access it via the `agent` property:

```python
def after_transition(self, event, state):
    self.agent.status.jobs_processed += 1
```

If the agent has been garbage-collected, accessing `self.agent` raises `RuntimeError("Agent has been deactivated")`.

## `BasicAgentStateMachine`

`BasicAgentStateMachine` is a ready-to-use concrete implementation included in the library. It relies entirely on the metaclass-injected defaults and implements `after_transition` to publish transitions when the publishing component is a `StateAgentPublishing`:

```python
from agentspype.agent.state_machine import BasicAgentStateMachine

class SimpleAgent(Agent):
    definition = AgentDefinition(
        state_machine_class=BasicAgentStateMachine,
        events_publishing_class=StateAgentPublishing,
        ...
    )
```

## Condition: `should_stop()`

`AgentStateMachine` exposes a `should_stop()` method that returns `True` after `on_stop()` has been called. You can use this as a polling condition in processing loops:

```python
while not agent.machine.should_stop():
    agent.machine.pick_job()
    process(...)
    agent.machine.finish_job()
```

## State Machine Independence

Each agent type maintains its own state machine class. Multiple agent types with different state machines coexist without any shared state:

```python
class MachineA(AgentStateMachine):
    starting = State("Starting", initial=True)
    a_state = State("A")
    end = State("End", final=True)
    go = starting.to(a_state)
    stop = a_state.to(end)
    def after_transition(self, event, state): pass


class MachineB(AgentStateMachine):
    starting = State("Starting", initial=True)
    b_state = State("B")
    end = State("End", final=True)
    go = starting.to(b_state)
    stop = b_state.to(end)
    def after_transition(self, event, state): pass


agent_a = AgentA({})  # uses MachineA
agent_b = AgentB({})  # uses MachineB

# Transitions on agent_a don't affect agent_b
agent_a.machine.go()
assert agent_a.machine.current_state == agent_a.machine.a_state
assert agent_b.machine.current_state == agent_b.machine.starting
```
