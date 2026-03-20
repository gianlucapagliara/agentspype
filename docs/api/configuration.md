# Configuration

**Module:** `agentspype.agent.configuration`, `agentspype.agent.definition`, `agentspype.agent.status`

---

## `AgentConfiguration`

**Module:** `agentspype.agent.configuration`

Base Pydantic model for agent configuration. All agent configuration classes must inherit from this.

```python
from agentspype.agent.configuration import AgentConfiguration
```

### Model Config

```python
model_config = ConfigDict(frozen=False, arbitrary_types_allowed=False)
```

- `frozen=False` — fields are mutable after construction.
- `arbitrary_types_allowed=False` — only types natively supported by Pydantic are allowed as field types.

### Subclassing

```python
class WorkerConfiguration(AgentConfiguration):
    worker_id: str
    max_retries: int = 3
    timeout_seconds: float = 30.0
    tags: list[str] = []
```

### Usage

Agents accept either an instance or a plain dict:

```python
agent = WorkerAgent({"worker_id": "w-001"})          # dict
agent = WorkerAgent(WorkerConfiguration(worker_id="w-001"))  # instance

agent.configuration.worker_id    # "w-001"
agent.configuration.max_retries  # 3 (default)
```

---

## `AgentDefinition`

**Module:** `agentspype.agent.definition`

Frozen Pydantic model that binds together all five component classes for an agent type. Declared as a class variable on `Agent` subclasses.

```python
from agentspype.agent.definition import AgentDefinition
```

### Model Config

```python
model_config = ConfigDict(frozen=True)
```

Frozen — the definition cannot be modified after creation.

### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `state_machine_class` | `type[AgentStateMachine]` | Yes | FSM class instantiated per agent |
| `events_listening_class` | `type[AgentListening]` | Yes | Event listener class instantiated per agent |
| `events_publishing_class` | `type[AgentPublishing]` | Yes | Event publisher class instantiated per agent |
| `configuration_class` | `type[AgentConfiguration]` | Yes | Used to validate dict configs and by Agency factory |
| `status_class` | `type[AgentStatus]` | Yes | Status class instantiated per agent |

### Usage

```python
class MyAgent(Agent):
    definition = AgentDefinition(
        state_machine_class=MyStateMachine,
        events_listening_class=MyListening,
        events_publishing_class=MyPublishing,
        configuration_class=MyConfiguration,
        status_class=MyStatus,
    )
```

The definition is evaluated once at class-definition time. All five fields are required; omitting any raises a Pydantic `ValidationError`.

### Accessing the Definition

```python
MyAgent.definition.state_machine_class    # MyStateMachine
MyAgent.definition.configuration_class    # MyConfiguration
```

---

## `AgentStatus`

**Module:** `agentspype.agent.status`

Base Pydantic model for agent runtime status. Intended to be updated during agent execution.

```python
from agentspype.agent.status import AgentStatus
```

### Model Config

```python
model_config = ConfigDict()
```

Default Pydantic config — mutable, no special restrictions.

### Subclassing

```python
class WorkerStatus(AgentStatus):
    jobs_processed: int = 0
    errors_encountered: int = 0
    last_job_id: str | None = None
    is_busy: bool = False
```

### Usage

Status is accessible via `agent.status` and updated directly:

```python
agent.status.jobs_processed += 1
agent.status.is_busy = True
agent.status.last_job_id = "job-99"
```

Status can be updated from state machine hooks:

```python
class WorkerStateMachine(AgentStateMachine):
    def on_enter_processing(self):
        self.agent.status.is_busy = True

    def on_exit_processing(self):
        self.agent.status.is_busy = False
        self.agent.status.jobs_processed += 1
```

### Serialization

Because `AgentStatus` is a Pydantic model, it supports `.model_dump()` and `.model_dump_json()`:

```python
snapshot = agent.status.model_dump()
# {"jobs_processed": 5, "is_busy": False, ...}
```
