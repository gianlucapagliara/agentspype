# Events

AgentsPype builds on [eventspype](https://github.com/gianlucapagliara/eventspype) for event pub/sub. Two component classes — `AgentListening` and `AgentPublishing` — wrap eventspype's `MultiSubscriber` and `MultiPublisher` respectively and integrate them into the agent lifecycle.

## Overview

- **Publishing** — An agent declares typed event publications as class attributes. It calls `publish()` to broadcast an event to all current subscribers.
- **Listening** — An agent subscribes to events from other publishers. Subscriptions are set up in `subscribe()` and torn down in `unsubscribe()`.
- **State transition events** — `StateAgentPublishing` is a specialised publisher that broadcasts a `StateMachineEvent` after every state machine transition.

Agents never hold direct references to each other. Communication goes through the eventspype broker, keeping components loosely coupled.

## Event Publishing

### `AgentPublishing`

`AgentPublishing` extends `eventspype.MultiPublisher`. To publish events, declare `EventPublication` instances as class attributes and call `self.publish()`.

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

The `event_tag` is an enum value used by subscribers to filter events. The `event_class` is the data class type that carries event payload.

### Accessing the Agent from Publishing

`AgentPublishing` holds a `weakref` to the agent and exposes it via the `agent` property:

```python
def publish_job_completed(self, job_id, result):
    self.agent.logger().info(f"Publishing job_completed for {job_id}")
    self.publish(self.job_completed, self.JobCompletedEvent(job_id, result))
```

### Discovering Published Events

`AgentPublishing.get_event_definitions()` returns all `EventPublication` instances declared on the class (used by the visualization system):

```python
definitions = WorkerPublishing.get_event_definitions()
# {'job_completed': <EventPublication>, 'job_failed': <EventPublication>}
```

## State Transition Events with `StateAgentPublishing`

`StateAgentPublishing` is a subclass of `AgentPublishing` that adds one pre-built publication: `sm_transition_event`. It broadcasts a `StateMachineEvent` dataclass containing the event name and new state after every transition.

```python
from agentspype.agent.publishing import StateAgentPublishing

# StateAgentPublishing.Events.StateMachineTransition  — the event tag
# StateAgentPublishing.StateMachineEvent(event, new_state)  — the data class
# StateAgentPublishing.sm_transition_event  — the EventPublication
```

To broadcast transitions, call `publish_transition` from the state machine's `after_transition` hook:

```python
class MyStateMachine(AgentStateMachine):
    ...
    def after_transition(self, event, state):
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
```

Other agents can subscribe to `StateAgentPublishing.Events.StateMachineTransition` to react to state changes in this agent.

## Event Listening

### `AgentListening`

`AgentListening` extends `eventspype.MultiSubscriber`. You must implement two abstract methods:

- `subscribe()` — called by `AgentStateMachine.on_start()` when the `start` transition fires.
- `unsubscribe()` — called by `Agent.teardown()` when the agent shuts down.

```python
from agentspype.agent.listening import AgentListening


class WorkerListening(AgentListening):

    def subscribe(self):
        # Register subscriptions here using eventspype's API.
        # For example (API depends on eventspype version):
        # self.add_subscription(
        #     publisher=some_publisher,
        #     event_tag=SomePublisher.Events.SomeEvent,
        #     callback=self.on_some_event,
        # )
        self.agent.logger().debug("Subscribed to events")

    def unsubscribe(self):
        # Remove all subscriptions
        # self.remove_all_subscriptions()
        self.agent.logger().debug("Unsubscribed from events")

    # --- Callbacks ---

    def on_job_available(self, event_data) -> None:
        """Called when a job becomes available."""
        self.agent.machine.pick_job()

    def on_shutdown_requested(self, event_data) -> None:
        """Called when a shutdown is requested."""
        self.agent.machine.safe_stop()
```

### Accessing the Agent from Listening

Like publishing, `AgentListening` exposes the agent via a `weakref`-backed property:

```python
def on_job_available(self, event_data):
    self.agent.status.jobs_pending += 1
    self.agent.machine.pick_job()
```

### Discovering Subscriptions

`AgentListening.get_event_definitions()` returns a dict of public, non-dunder, non-abstract callable methods. This is used by the visualization system to enumerate callbacks. The returned dict includes `event_tag` and `publisher_class` attributes if the methods carry them (set by the eventspype subscription decorators).

## Lifecycle Integration

The listening/publishing lifecycle is tied to the state machine:

```
machine.start()
  └── on_start() → listening.subscribe()

machine.stop() or machine entering a final state
  └── teardown() → listening.unsubscribe()
```

This means an agent only receives events while it is running (between `start` and `stop`). Before `start()` is called and after `stop()` fires, no callbacks will be invoked even if events are published.

## Example: Two Communicating Agents

```python
# Producer publishes work items
class ProducerPublishing(StateAgentPublishing):
    class Events(Enum):
        WorkAvailable = "work_available"

    @dataclass
    class WorkEvent:
        item: str

    work_available = EventPublication(
        event_tag=Events.WorkAvailable,
        event_class=WorkEvent,
    )

    def publish_work(self, item: str) -> None:
        self.publish(self.work_available, self.WorkEvent(item=item))


# Consumer listens to the producer
class ConsumerListening(AgentListening):
    def subscribe(self):
        # Subscribe to the producer's work_available event
        # self.add_subscription(producer_instance.publishing, ...)
        pass

    def unsubscribe(self):
        pass

    def on_work_available(self, event: ProducerPublishing.WorkEvent) -> None:
        self.agent.logger().info(f"Received work: {event.item}")
        self.agent.machine.pick_job()
```
