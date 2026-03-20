# Agency

`Agency` is a class-level registry that tracks all live agent instances and maps agent classes to their configuration classes. It provides a factory method for creating agents from configuration objects when the concrete agent type is not known at the call site.

## What Agency Does

- Maintains a list of all currently active (registered) agents.
- Keeps a bidirectional mapping between agent classes and their configuration classes.
- Provides a factory: given a configuration object, look up the corresponding agent class and instantiate it.

`Agency` uses only `@classmethod` methods; there is no instance to create.

## Automatic Registration

Agents register and deregister themselves automatically:

- **Registration** happens at the end of `Agent.__init__()`, before `initialize()` is called.
- **Deregistration** happens inside `Agent.teardown()`, which is triggered by the state machine entering a final state or by calling `teardown()` directly.

```python
from agentspype.agency import Agency

agent = MyAgent({})
assert agent in Agency.get_active_agents()

agent.machine.stop()  # or agent.teardown()
assert agent not in Agency.get_active_agents()
```

## Methods

### `register_agent(agent)`

Adds an agent to `initialized_agents`. Idempotent — registering an already-registered agent is a no-op.

```python
Agency.register_agent(agent)
```

### `deregister_agent(agent)`

Removes an agent from `initialized_agents` and schedules async cleanup (moves the agent to a temporary `_deactivating_agents` list, then removes it after yielding to the event loop). Idempotent.

```python
Agency.deregister_agent(agent)
```

### `get_active_agents()`

Returns a snapshot (list copy) of all currently registered agents.

```python
active = Agency.get_active_agents()
for agent in active:
    print(agent.complete_name)
```

### `register_agent_class(agent_class)`

Associates an agent class with its configuration class (taken from `agent_class.definition.configuration_class`). Idempotent. Must be called before `get_agent_from_configuration` can work for this class.

```python
Agency.register_agent_class(WorkerAgent)
```

### `deregister_agent_class(agent_class)`

Removes the agent class mapping.

```python
Agency.deregister_agent_class(WorkerAgent)
```

### `get_agent_from_configuration(configuration)`

Factory method. Looks up the agent class for the given configuration type and calls it with `configuration`. Raises `ValueError` if no agent class is registered for that configuration type.

```python
Agency.register_agent_class(WorkerAgent)

config = WorkerConfiguration(worker_id="w-001")
agent = Agency.get_agent_from_configuration(config)
# equivalent to: WorkerAgent(config)
```

## Factory Pattern

The factory is useful when a component receives a configuration object but does not know (or should not care) which concrete agent class to instantiate. Register all agent classes once at startup, then dispatch purely on configuration type:

```python
# At application startup
Agency.register_agent_class(WorkerAgent)
Agency.register_agent_class(MonitorAgent)
Agency.register_agent_class(CoordinatorAgent)

# Later, in a dispatcher that receives raw config objects
def spawn_agent(config):
    return Agency.get_agent_from_configuration(config)

worker_config = WorkerConfiguration(worker_id="w-001")
agent = spawn_agent(worker_config)  # returns a WorkerAgent
```

## Observing Active Agents

You can query the live agent list at any time:

```python
active = Agency.get_active_agents()
print(f"{len(active)} agents running")
for a in active:
    print(f"  {a.complete_name} — state: {a.machine.current_state.id}")
```

## Cleanup and Deactivation

When `deregister_agent` is called, the agent is moved to an internal `_deactivating_agents` list and scheduled for removal on the next event loop tick (`asyncio.ensure_future`). This brief window prevents the agent from being garbage-collected mid-teardown while still allowing subscribers to finish processing any in-flight events.

!!! note
    `Agency` uses a plain list as `initialized_agents`. In multi-threaded code, concurrent registration and deregistration should be protected by a lock. AgentsPype is primarily designed for single-threaded async applications.

## Resetting Agency State

In tests, you can clear the registry between test cases:

```python
import pytest
from agentspype.agency import Agency

@pytest.fixture(autouse=True)
def clear_agency():
    Agency.initialized_agents.clear()
    yield
    Agency.initialized_agents.clear()
```
