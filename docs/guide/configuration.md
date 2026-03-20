# Configuration, Definitions, and Status

AgentsPype uses three Pydantic-based structures to describe and configure agents: `AgentConfiguration`, `AgentDefinition`, and `AgentStatus`.

## AgentConfiguration

`AgentConfiguration` is the base class for all agent configuration models. It is a Pydantic `BaseModel` with `frozen=False` (mutable) and `arbitrary_types_allowed=False`.

### Defining Configuration

Add fields using standard Pydantic syntax:

```python
from agentspype.agent.configuration import AgentConfiguration


class WorkerConfiguration(AgentConfiguration):
    worker_id: str
    max_retries: int = 3
    queue_name: str = "default"
    timeout_seconds: float = 30.0
```

### Creating Configuration Instances

Configuration can be created directly from a Pydantic model instance or from a plain dict. `Agent.__init__` accepts both:

```python
# From a dict — Pydantic validates and coerces values
agent = WorkerAgent({"worker_id": "w-001", "max_retries": 5})

# From a configuration object
config = WorkerConfiguration(worker_id="w-002")
agent = WorkerAgent(config)
```

When a dict is passed, the agent calls `configuration_class(**configuration)` internally.

### Accessing Configuration

```python
agent.configuration.worker_id  # "w-001"
agent.configuration.max_retries  # 3
```

Configuration is accessible at any time via `agent.configuration`. Because the model is mutable (`frozen=False`), fields can be updated at runtime, though this is typically not recommended.

### Validation

Because `AgentConfiguration` is a Pydantic `BaseModel`, all fields are validated on construction. Invalid data raises a `ValidationError`:

```python
WorkerConfiguration(worker_id=123)  # OK — coerced to str
WorkerConfiguration()               # ValidationError — worker_id is required
WorkerConfiguration(worker_id="w", max_retries="not_an_int")  # ValidationError
```

## AgentDefinition

`AgentDefinition` is an **immutable** (frozen) Pydantic model that binds the five component classes for an agent type. It is declared as a class variable on the `Agent` subclass.

```python
from agentspype.agent.definition import AgentDefinition


class MyAgent(Agent):
    definition = AgentDefinition(
        state_machine_class=MyStateMachine,
        events_listening_class=MyListening,
        events_publishing_class=MyPublishing,
        configuration_class=MyConfiguration,
        status_class=MyStatus,
    )
```

### Fields

| Field | Type | Description |
|---|---|---|
| `state_machine_class` | `type[AgentStateMachine]` | The FSM class to instantiate |
| `events_listening_class` | `type[AgentListening]` | The event listener class to instantiate |
| `events_publishing_class` | `type[AgentPublishing]` | The event publisher class to instantiate |
| `configuration_class` | `type[AgentConfiguration]` | Used to parse dicts and by Agency factory |
| `status_class` | `type[AgentStatus]` | The status class to instantiate |

All five fields are required. The definition is validated once when the class is defined. Because the model is frozen, it cannot be modified after creation.

### How Agent Uses the Definition

During `Agent.__init__`, the definition is read to construct each component:

```python
self._events_publishing = self.definition.events_publishing_class(self)
self._events_listening = self.definition.events_listening_class(self)
self._state_machine = self.definition.state_machine_class(self)
self._status = self.definition.status_class()
```

The `configuration_class` is also used by `Agency.register_agent_class` to build the class-to-configuration bidirectional mapping.

### Shared Definitions

Multiple agent classes can share component classes (e.g., the same configuration or status type), but each `AgentDefinition` is its own frozen instance:

```python
SHARED_CONFIG = AgentConfiguration  # use the base class as-is

class AgentA(Agent):
    definition = AgentDefinition(
        configuration_class=SHARED_CONFIG,
        state_machine_class=MachineA,
        ...
    )

class AgentB(Agent):
    definition = AgentDefinition(
        configuration_class=SHARED_CONFIG,  # same config class
        state_machine_class=MachineB,
        ...
    )
```

!!! warning
    If two agent classes share the same `configuration_class`, `Agency.register_agent_class` can only map one of them to that configuration type (the bidirectional mapping enforces a 1-to-1 relationship). Use distinct configuration subclasses if you need the Agency factory to distinguish between them.

## AgentStatus

`AgentStatus` is the base class for runtime status models. Unlike configuration (which is set at startup), status is intended to be updated during agent execution.

```python
from agentspype.agent.status import AgentStatus


class WorkerStatus(AgentStatus):
    jobs_processed: int = 0
    errors_encountered: int = 0
    last_job_id: str | None = None
    is_busy: bool = False
```

### Updating Status

The status object is accessible via `agent.status`. Update fields directly:

```python
agent.status.jobs_processed += 1
agent.status.last_job_id = "job-42"
agent.status.is_busy = False
```

### Using Status in State Machine Hooks

```python
class WorkerStateMachine(AgentStateMachine):
    ...
    def on_enter_processing(self):
        self.agent.status.is_busy = True

    def on_exit_processing(self):
        self.agent.status.is_busy = False
        self.agent.status.jobs_processed += 1
```

### Default Configuration

`AgentStatus` uses `ConfigDict()` with no additional constraints. Extend it freely with any Pydantic-compatible field types.

## Combining All Three

A complete example showing all three structures:

```python
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.status import AgentStatus
from agentspype.agent.agent import Agent
from agentspype.agent.state_machine import BasicAgentStateMachine
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing


class PipelineConfiguration(AgentConfiguration):
    pipeline_name: str
    batch_size: int = 100
    dry_run: bool = False


class PipelineStatus(AgentStatus):
    batches_processed: int = 0
    records_total: int = 0
    started_at: str | None = None


class PipelineListening(AgentListening):
    def subscribe(self): pass
    def unsubscribe(self): pass


class PipelineAgent(Agent):
    definition = AgentDefinition(
        configuration_class=PipelineConfiguration,
        events_publishing_class=StateAgentPublishing,
        events_listening_class=PipelineListening,
        state_machine_class=BasicAgentStateMachine,
        status_class=PipelineStatus,
    )

    def initialize(self):
        self.logger().info(
            f"Pipeline '{self.configuration.pipeline_name}' ready "
            f"(batch_size={self.configuration.batch_size})"
        )


agent = PipelineAgent({
    "pipeline_name": "etl-prod",
    "batch_size": 500,
})
agent.status.started_at = "2026-01-01T00:00:00Z"
agent.machine.start()
```
