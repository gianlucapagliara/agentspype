# Agency

**Module:** `agentspype.agency`

## `Agency`

Class-level registry for agent instances and agent class mappings. All methods are class methods; `Agency` is never instantiated.

### Class Variables

| Name | Type | Description |
|---|---|---|
| `initialized_agents` | `list[Agent]` | All currently registered (active) agent instances |
| `_deactivating_agents` | `list[Agent]` | Agents pending async cleanup after deregistration |
| `_agent_to_configuration` | `bidict[type[Agent], type[AgentConfiguration]]` | Bidirectional map between agent classes and configuration classes |
| `_logger` | `logging.Logger` | Logger named `agentspype.agency` |

### Methods

#### `register_agent(agent)` *(classmethod)*

```python
Agency.register_agent(agent: Agent) -> None
```

Adds `agent` to `initialized_agents`. Idempotent — calling with an already-registered agent is a no-op. Logs at INFO level.

#### `deregister_agent(agent)` *(classmethod)*

```python
Agency.deregister_agent(agent: Agent) -> None
```

Removes `agent` from `initialized_agents` and appends it to `_deactivating_agents`. Schedules an `asyncio` coroutine to remove it from `_deactivating_agents` after the current event loop tick. Idempotent. Logs at INFO level.

The brief stay in `_deactivating_agents` prevents premature garbage-collection while teardown completes.

#### `get_active_agents()` *(classmethod)*

```python
Agency.get_active_agents() -> list[Agent]
```

Returns a shallow copy of `initialized_agents`. Modifying the returned list does not affect the registry.

#### `register_agent_class(agent_class)` *(classmethod)*

```python
Agency.register_agent_class(agent_class: type[Agent]) -> None
```

Maps `agent_class` to `agent_class.definition.configuration_class` in the bidirectional dict. Idempotent. Required before `get_agent_from_configuration` can work for this class.

#### `deregister_agent_class(agent_class)` *(classmethod)*

```python
Agency.deregister_agent_class(agent_class: type[Agent]) -> None
```

Removes the mapping for `agent_class`. Idempotent.

#### `resolve_by_name(class_name)` *(classmethod)*

```python
Agency.resolve_by_name(class_name: str) -> type[Agent]
```

Resolves a registered agent class by its `__name__`.

**Raises:**
- `KeyError` — if no registered agent class matches `class_name`.
- `ValueError` — if multiple registered agent classes share the same `class_name`.

```python
Agency.register_agent_class(WorkerAgent)
agent_cls = Agency.resolve_by_name("WorkerAgent")
# agent_cls is WorkerAgent
```

#### `get_registered_agent_classes()` *(classmethod)*

```python
Agency.get_registered_agent_classes() -> dict[str, type[Agent]]
```

Returns all registered agent classes keyed by `__name__`.

```python
Agency.register_agent_class(WorkerAgent)
classes = Agency.get_registered_agent_classes()
# {"WorkerAgent": WorkerAgent}
```

#### `get_agent_from_configuration(configuration)` *(classmethod)*

```python
Agency.get_agent_from_configuration(
    configuration: AgentConfiguration
) -> Agent
```

Factory method. Looks up the agent class registered for `type(configuration)` and calls `agent_class(configuration)`.

**Raises:** `ValueError` — if no agent class is registered for `type(configuration)`.

```python
Agency.register_agent_class(WorkerAgent)
config = WorkerConfiguration(worker_id="w-001")
agent = Agency.get_agent_from_configuration(config)
# equivalent to WorkerAgent(config)
```

### Usage Patterns

#### Monitoring Active Agents

```python
active = Agency.get_active_agents()
print(f"{len(active)} agents running")
```

#### Factory Dispatch

```python
# Register all known agent types at startup
Agency.register_agent_class(WorkerAgent)
Agency.register_agent_class(MonitorAgent)

# Dispatch on config type later
def spawn(config):
    return Agency.get_agent_from_configuration(config)
```

#### Test Setup

```python
@pytest.fixture(autouse=True)
def clear_agency():
    Agency.initialized_agents.clear()
    yield
    Agency.initialized_agents.clear()
```
