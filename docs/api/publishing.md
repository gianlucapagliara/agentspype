# Publishing

**Module:** `agentspype.agent.publishing`

## `AgentPublishing`

Base class for agent event publishers. Extends `eventspype.pub.multipublisher.MultiPublisher` with agent integration.

### Constructor

```python
AgentPublishing(agent: Agent) -> None
```

Stores a `weakref` to `agent` and calls `super().__init__()`.

### Properties

#### `agent -> Agent`

Resolves the weakref to the owning agent. Raises `RuntimeError("Agent has been deactivated")` if the agent has been garbage-collected.

### Methods

#### `get_event_definitions() -> dict[str, EventPublication]` *(optional classmethod)*

Not defined on the base class. Subclasses can implement this to support the visualization system. The `PublishingVisualization` checks for its presence with `hasattr()` and, if found, calls it to enumerate `EventPublication` attributes.

A typical implementation scans class attributes and returns those that are `EventPublication` instances:

```python
WorkerPublishing.get_event_definitions()
# {"job_completed": <EventPublication>, "job_failed": <EventPublication>}
```

### Declaring Events

Declare `EventPublication` instances as class attributes. Each publication needs an `event_tag` (an enum value) and an `event_class` (a data class or any type).

```python
from enum import Enum
from dataclasses import dataclass
from eventspype.pub.publication import EventPublication
from agentspype.agent.publishing import AgentPublishing


class WorkerPublishing(AgentPublishing):

    class Events(Enum):
        JobCompleted = "job_completed"
        JobFailed = "job_failed"

    @dataclass
    class JobCompletedEvent:
        job_id: str
        result: str

    @dataclass
    class JobFailedEvent:
        job_id: str
        error: str

    job_completed = EventPublication(
        event_tag=Events.JobCompleted,
        event_class=JobCompletedEvent,
    )
    job_failed = EventPublication(
        event_tag=Events.JobFailed,
        event_class=JobFailedEvent,
    )

    def publish_job_completed(self, job_id: str, result: str) -> None:
        self.publish(
            self.job_completed,
            self.JobCompletedEvent(job_id=job_id, result=result),
        )

    def publish_job_failed(self, job_id: str, error: str) -> None:
        self.publish(
            self.job_failed,
            self.JobFailedEvent(job_id=job_id, error=error),
        )
```

---

## `StateAgentPublishing`

**Module:** `agentspype.agent.publishing`

Extends `AgentPublishing` with a pre-built state machine transition event. Use this (or a subclass of it) when you want the agent to broadcast its state transitions.

### Nested Types

#### `Events` *(Enum)*

```python
class Events(Enum):
    StateMachineTransition = "sm_transition_event"
```

#### `StateMachineEvent` *(dataclass)*

```python
@dataclass
class StateMachineEvent:
    event: Any       # the transition name (str)
    new_state: State # the state entered
```

### Class Attributes

#### `sm_transition_event`

```python
sm_transition_event = EventPublication(
    event_tag=Events.StateMachineTransition,
    event_class=StateMachineEvent,
)
```

An `EventPublication` that subscribers can use to listen for state transitions from this agent.

### Methods

#### `publish_transition(event, state) -> None`

```python
StateAgentPublishing.publish_transition(
    event: Any,
    state: State,
) -> None
```

Publishes a `StateMachineEvent(event, state)` via `sm_transition_event`. Call this from the state machine's `after_transition` hook:

```python
class MyStateMachine(AgentStateMachine):
    def after_transition(self, event, state):
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
```

### Extending `StateAgentPublishing`

You can add custom events on top of the built-in transition event:

```python
class WorkerPublishing(StateAgentPublishing):
    class Events(Enum):
        # Keep the inherited transition event tag
        StateMachineTransition = "sm_transition_event"
        # Add custom events
        JobCompleted = "job_completed"

    @dataclass
    class JobCompletedEvent:
        job_id: str

    job_completed = EventPublication(
        event_tag=Events.JobCompleted,
        event_class=JobCompletedEvent,
    )

    # Inherit sm_transition_event and publish_transition from StateAgentPublishing
    sm_transition_event = StateAgentPublishing.sm_transition_event

    def publish_job_completed(self, job_id: str) -> None:
        self.publish(self.job_completed, self.JobCompletedEvent(job_id=job_id))
```
