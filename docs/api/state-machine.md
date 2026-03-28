# State Machine

**Module:** `agentspype.agent.state_machine`

## `AgentStateMachine`

Base class for all agent state machines. Extends `agentspype.fsm.StateMachine` with agent integration: a `weakref` back to the owning agent, automatic subscription on start, automatic teardown on entering a final state, and default state/transition injection via `StateMachineMeta`.

### Class Variables

The following class variables are injected by `StateMachineMeta` into every concrete subclass that does not define them:

| Name | Type | Description |
|---|---|---|
| `starting` | `State(initial=True)` | Initial state |
| `idle` | `State` | Default idle/ready state |
| `end` | `State(final=True)` | Terminal state |
| `start` | `TransitionList` | `starting → idle` |
| `stop` | `TransitionList` | `starting → end | idle → end` |

### Constructor

```python
AgentStateMachine(agent: Agent) -> None
```

Stores a `weakref` to `agent` and sets `_should_stop = False`. Calls `super().__init__()` from `StateMachine`.

### Properties

#### `agent -> Agent`

Resolves the weakref. Raises `RuntimeError("Agent has been deactivated")` if the agent has been garbage-collected.

### Built-in Hooks

#### `on_enter_end() -> None`

Called automatically when the machine enters the `end` state. Calls `self.agent.teardown()`.

#### `before_transition(event, state, source, target) -> None`

Called before every transition. If `source != target`, logs a DEBUG message:
`[ClassName:StateMachine] (event) source -> target`

Override to add pre-transition guards or validation.

#### `after_transition(event, state) -> None` *(abstract)*

Called after every successful transition. **Must be implemented in every concrete subclass.** Typical implementation:

```python
def after_transition(self, event, state):
    if isinstance(self.agent.publishing, StateAgentPublishing):
        self.agent.publishing.publish_transition(event, state)
```

#### `on_start() -> None`

Called when the `start` transition fires. Calls `self.agent.listening.subscribe()`.

#### `on_stop() -> None`

Called when the `stop` transition fires. Sets `_should_stop = True`.

### Condition

#### `should_stop() -> bool`

Returns `True` after `on_stop()` has been called. Useful for polling loops.

### Convenience Methods

#### `safe_start() -> None`

Fires `start(f=True)` only if the machine is in an initial state. No-op otherwise.

```python
agent.machine.safe_start()  # safe to call multiple times
```

#### `safe_stop() -> None`

Fires `stop(f=True)` only if the machine is not in a final state. No-op otherwise.

```python
agent.machine.safe_stop()  # safe to call multiple times
```

---

## `StateMachineMeta`

**Module:** `agentspype.fsm.machine`

Metaclass for `StateMachine`. Handles state collection, transition mapping, hook resolution, and state isolation across inheritance hierarchies.

### Behavior

When a new class is created using this metaclass (other than `AgentStateMachine` itself), the metaclass inspects the class namespace and injects missing defaults:

- If `starting` is not in the namespace, injects `State("Starting", initial=True)`.
- If `idle` is not in the namespace, injects `State("Idle")`.
- If `end` is not in the namespace, injects `State("End", final=True)`.
- If `start` is not in the namespace, injects `starting.to(idle)`.
- If `stop` is not in the namespace, injects `starting.to(end) | idle.to(end)`.

This happens at **class creation time** (before any instance is created), so the injected states and transitions behave identically to explicitly declared ones.

### Overriding Defaults

If you declare any of these names in your subclass, the metaclass will not overwrite them:

```python
class MyMachine(AgentStateMachine):
    # Override end — metaclass will not inject a default end state
    starting = State("Starting", initial=True)
    processing = State("Processing")
    done = State("Done", final=True)  # use 'done' instead of 'end'

    start = starting.to(processing)
    stop = processing.to(done)

    def after_transition(self, event, state): pass
```

---

## `BasicAgentStateMachine`

**Module:** `agentspype.agent.state_machine`

A ready-to-use concrete subclass of `AgentStateMachine`. Uses all metaclass-injected defaults and implements `after_transition` to publish transitions via `StateAgentPublishing`.

```python
class BasicAgentStateMachine(AgentStateMachine):
    def after_transition(self, event: str, state: State) -> None:
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
```

### States

- `starting` (initial)
- `idle`
- `end` (final)

### Transitions

- `start`: `starting → idle`
- `stop`: `starting → end | idle → end`

### Usage

```python
from agentspype.agent.state_machine import BasicAgentStateMachine

class SimpleAgent(Agent):
    definition = AgentDefinition(
        state_machine_class=BasicAgentStateMachine,
        events_publishing_class=StateAgentPublishing,
        ...
    )
```
