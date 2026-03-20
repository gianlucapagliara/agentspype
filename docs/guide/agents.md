# Agents

The `Agent` class is the central abstraction in AgentsPype. An agent owns exactly one instance of each of its components — a state machine, an event listener, an event publisher, a configuration object, and a status object. These are assembled via an immutable `AgentDefinition` declared on the class.

## Lifecycle

```
__init__()
  ├── build configuration
  ├── build publishing, listening, state_machine, status
  ├── Agency.register_agent(self)
  └── initialize()   ← override this hook

machine.start()
  └── on_start()     ← calls listening.subscribe()

...agent is running...

machine.stop()  or  teardown()
  ├── listening.unsubscribe()
  └── Agency.deregister_agent(self)

machine.on_enter_end()
  └── calls teardown()
```

### `initialize()`

Override `initialize()` to run code after all components have been wired together but before the agent starts processing events. The default implementation is a no-op.

```python
class MyAgent(Agent):
    definition = ...

    def initialize(self):
        super().initialize()
        self.logger().info("Agent ready")
        # load initial data, connect to resources, etc.
```

### `teardown()`

`teardown()` unsubscribes all event listeners and deregisters the agent from `Agency`. It is safe to call multiple times (subsequent calls after deregistration are silently ignored by `Agency`).

`teardown()` is called automatically:
- When the state machine enters any `final=True` state (via `on_enter_end`).
- When the agent is garbage-collected (via `__del__`, with exceptions suppressed).

You can also trigger it explicitly:

```python
agent.teardown()
```

### `clone()`

`clone()` creates a new agent of the same type with a copy of the current configuration. The new agent goes through the full initialization sequence independently.

```python
original = MyAgent({"worker_id": "w-001"})
copy = original.clone()
# copy is a brand-new MyAgent with the same configuration
```

## Defining an Agent

### The Definition

`AgentDefinition` is a frozen Pydantic model that wires together all five component classes:

```python
from agentspype.agent.definition import AgentDefinition

class MyAgent(Agent):
    definition = AgentDefinition(
        configuration_class=MyConfiguration,
        events_publishing_class=MyPublishing,
        events_listening_class=MyListening,
        state_machine_class=MyStateMachine,
        status_class=MyStatus,
    )
```

All five fields are required. This declaration is evaluated once at class definition time. Because `AgentDefinition` is frozen, it cannot be changed at runtime.

### Minimal Agent

If you do not need custom configuration, status, listening, or publishing, you can use the base classes directly:

```python
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import BasicAgentStateMachine
from agentspype.agent.status import AgentStatus


class NoOpListening(AgentListening):
    def subscribe(self): pass
    def unsubscribe(self): pass


class MinimalAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=StateAgentPublishing,
        events_listening_class=NoOpListening,
        state_machine_class=BasicAgentStateMachine,
        status_class=AgentStatus,
    )
```

## Initialization from Dict

Agents accept either an `AgentConfiguration` instance or a plain `dict`. When a dict is passed, it is forwarded to `configuration_class(**configuration)` — Pydantic validates and coerces the values.

```python
agent1 = MyAgent({"worker_id": "w-001", "max_retries": 5})
agent2 = MyAgent(MyConfiguration(worker_id="w-002"))
```

## Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | `AID:<id>|<ClassName>` — unique identifier including the Python object id |
| `complete_name` | `str` | Adds `PID:<parent_id>|` prefix for child agents |
| `configuration` | `AgentConfiguration` | The agent's configuration instance |
| `machine` | `AgentStateMachine` | The agent's state machine instance |
| `listening` | `AgentListening` | The agent's event listening instance |
| `publishing` | `AgentPublishing` | The agent's event publishing instance |
| `status` | `AgentStatus` | The agent's runtime status instance |
| `parent_id` | `int | None` | Optional parent agent id (for hierarchical agents) |

## Parent–Child Relationships

Agents can optionally track a parent by passing `parent_id` at construction time. Once set, `parent_id` cannot be changed to another value (it can only be cleared back to `None`).

```python
parent = MyAgent({})
child = MyAgent({}, parent_id=id(parent))
print(child.complete_name)  # PID:<parent_id>|AID:<child_id>|MyAgent
```

## Logging

Each agent class lazily initializes a logger named after the class:

```python
MyAgent.logger().info("Class-level log")

agent = MyAgent({})
agent.logger().debug("Instance log — same logger")
```

## Visualization

Every agent provides four visualization methods. These are documented in detail in the [Visualization guide](visualization.md).

```python
agent.visualize(save_file=True)              # full diagram
agent.visualize_state_machine(save_file=True)
agent.visualize_publishing(save_file=True)
agent.visualize_listening(save_file=True)
agent.create_all_diagrams(save_files=True)  # all four at once
```
