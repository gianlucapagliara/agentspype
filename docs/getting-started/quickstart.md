# Quick Start

This guide walks through building a functional agent step by step. By the end you will have an agent with a custom state machine that publishes transition events.

## Step 1 — Define a Configuration

Configuration is a Pydantic `BaseModel`. Add fields for anything the agent needs at startup.

```python
from agentspype.agent.configuration import AgentConfiguration


class WorkerConfiguration(AgentConfiguration):
    worker_id: str
    max_retries: int = 3
```

## Step 2 — Define a Status

Status tracks runtime information. It is mutable and updated by the agent during execution.

```python
from agentspype.agent.status import AgentStatus


class WorkerStatus(AgentStatus):
    jobs_processed: int = 0
    errors_encountered: int = 0
```

## Step 3 — Define a State Machine

Subclass `AgentStateMachine` and declare states and transitions as class attributes.

```python
from statemachine import State
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.publishing import StateAgentPublishing


class WorkerStateMachine(AgentStateMachine):
    # States
    starting = State("Starting", initial=True)
    idle = State("Idle")
    processing = State("Processing")
    end = State("End", final=True)

    # Transitions
    start = starting.to(idle)
    pick_job = idle.to(processing)
    finish_job = processing.to(idle)
    stop = starting.to(end) | idle.to(end) | processing.to(end)

    def after_transition(self, event, state):
        # Publish the transition so other agents can react
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)
```

The `after_transition` hook is called after every successful transition. Here it forwards the transition as an event to any subscribers.

!!! note
    `after_transition` is abstract in `AgentStateMachine`. Every concrete subclass must implement it.

## Step 4 — Define Event Listening

Subclass `AgentListening` and implement `subscribe` and `unsubscribe`. These are called automatically when the state machine enters and leaves the `idle` state (via the `on_start` / `teardown` hooks on the base class).

```python
from agentspype.agent.listening import AgentListening


class WorkerListening(AgentListening):
    def subscribe(self):
        # Register subscriptions using eventspype's subscription API
        # Example (requires an actual publisher to exist):
        # self.add_subscription(some_publisher, SomeEvent.Tag, self.on_job_received)
        self.agent.logger().info("WorkerListening: subscribed")

    def unsubscribe(self):
        # self.remove_all_subscriptions()
        self.agent.logger().info("WorkerListening: unsubscribed")

    def on_job_received(self, event_data):
        """Called when a job event is received."""
        self.agent.machine.pick_job()
```

## Step 5 — Define Event Publishing

For simple agents, `StateAgentPublishing` is often sufficient. It provides a single `sm_transition_event` publication that broadcasts every state transition.

For custom events, extend `AgentPublishing` (or `StateAgentPublishing`) and declare `EventPublication` attributes:

```python
from enum import Enum
from dataclasses import dataclass
from eventspype.pub.publication import EventPublication
from agentspype.agent.publishing import StateAgentPublishing


class WorkerPublishing(StateAgentPublishing):
    class Events(Enum):
        JobCompleted = "job_completed"

    @dataclass
    class JobCompletedEvent:
        job_id: str
        result: str

    job_completed = EventPublication(
        event_tag=Events.JobCompleted,
        event_class=JobCompletedEvent,
    )

    def publish_job_completed(self, job_id: str, result: str) -> None:
        self.publish(
            self.job_completed,
            self.JobCompletedEvent(job_id=job_id, result=result),
        )
```

## Step 6 — Assemble the Agent

Bind all components together in an `AgentDefinition` and declare it as a class variable on the `Agent` subclass.

```python
from agentspype.agent.agent import Agent
from agentspype.agent.definition import AgentDefinition


class WorkerAgent(Agent):
    definition = AgentDefinition(
        configuration_class=WorkerConfiguration,
        events_publishing_class=WorkerPublishing,
        events_listening_class=WorkerListening,
        state_machine_class=WorkerStateMachine,
        status_class=WorkerStatus,
    )

    def initialize(self):
        """Called at the end of __init__. Set up any initial state here."""
        super().initialize()
        self.logger().info(f"Worker {self.configuration.worker_id} initialized")
```

## Step 7 — Run the Agent

```python
# Create from a dict — fields are validated by Pydantic
agent = WorkerAgent({"worker_id": "w-001", "max_retries": 5})

# Or from a configuration object
config = WorkerConfiguration(worker_id="w-002")
agent2 = WorkerAgent(config)

# Drive the state machine
agent.machine.start()        # starting -> idle, triggers subscribe()
agent.machine.pick_job()     # idle -> processing
agent.machine.finish_job()   # processing -> idle
agent.machine.stop()         # idle -> end, triggers teardown()
```

## Step 8 — Visualize (Optional)

```python
# Create and save a comprehensive diagram
agent = WorkerAgent({"worker_id": "w-001"})
agent.visualize(save_file=True, output_dir=".diagrams")

# Or just the state machine
agent.visualize_state_machine(save_file=True)
```

Diagrams are saved as PNG files in the `output_dir` directory.

## Putting It All Together

```python
from statemachine import State
from dataclasses import dataclass
from enum import Enum

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from eventspype.pub.publication import EventPublication


class WorkerConfiguration(AgentConfiguration):
    worker_id: str
    max_retries: int = 3


class WorkerStatus(AgentStatus):
    jobs_processed: int = 0


class WorkerStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    idle = State("Idle")
    processing = State("Processing")
    end = State("End", final=True)

    start = starting.to(idle)
    pick_job = idle.to(processing)
    finish_job = processing.to(idle)
    stop = starting.to(end) | idle.to(end) | processing.to(end)

    def after_transition(self, event, state):
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)


class WorkerListening(AgentListening):
    def subscribe(self):
        pass

    def unsubscribe(self):
        pass


class WorkerAgent(Agent):
    definition = AgentDefinition(
        configuration_class=WorkerConfiguration,
        events_publishing_class=StateAgentPublishing,
        events_listening_class=WorkerListening,
        state_machine_class=WorkerStateMachine,
        status_class=WorkerStatus,
    )


agent = WorkerAgent({"worker_id": "w-001"})
agent.machine.start()
agent.machine.pick_job()
agent.machine.finish_job()
agent.machine.stop()
```
