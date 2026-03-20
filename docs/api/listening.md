# Listening

**Module:** `agentspype.agent.listening`

## `AgentListening`

Base class for agent event listeners. Extends `eventspype.sub.multisubscriber.MultiSubscriber` with agent integration.

### Constructor

```python
AgentListening(agent: Agent) -> None
```

Stores a `weakref` to `agent` and calls `super().__init__()`.

### Properties

#### `agent -> Agent`

Resolves the weakref to the owning agent. Raises `RuntimeError("Agent has been deactivated")` if the agent has been garbage-collected.

### Methods

#### `logger() -> logging.Logger`

Returns `self.agent.logger()` — the class-level logger of the owning agent.

#### `get_event_definitions() -> dict[str, Any]` *(classmethod)*

Returns a dict mapping method names to subscription metadata for all public, non-dunder, non-reserved callable methods on the class. Used by `ListeningVisualization` to enumerate callback methods.

The dict values are dicts with keys:
- `"callback"` — the method object
- `"callback_name"` — the method name
- `"event_tag"` — value of `getattr(method, "event_tag", None)` (set by eventspype decorators)
- `"publisher_class"` — value of `getattr(method, "publisher_class", None)` (set by eventspype decorators)

Reserved names excluded from the result: `subscribe`, `unsubscribe`, `logger`.

#### `subscribe() -> None` *(abstract)*

Called by `AgentStateMachine.on_start()` when the `start` transition fires. Implement to register event subscriptions using the eventspype API.

```python
def subscribe(self) -> None:
    # Example using eventspype subscription API
    # self.add_subscription(publisher, EventTag, self.on_event)
    pass
```

#### `unsubscribe() -> None` *(abstract)*

Called by `Agent.teardown()` when the agent shuts down. Implement to remove all subscriptions.

```python
def unsubscribe(self) -> None:
    # self.remove_all_subscriptions()
    pass
```

### Subclassing

Every `AgentListening` subclass must implement both `subscribe` and `unsubscribe`. A minimal no-op implementation:

```python
from agentspype.agent.listening import AgentListening


class NoOpListening(AgentListening):
    def subscribe(self) -> None:
        pass

    def unsubscribe(self) -> None:
        pass
```

A real implementation using eventspype:

```python
class WorkerListening(AgentListening):

    def subscribe(self) -> None:
        # subscribe to events from a publisher
        # self.add_subscription(
        #     publisher=some_publisher_instance,
        #     event_tag=SomePublisher.Events.SomeTag,
        #     callback=self.on_some_event,
        # )
        pass

    def unsubscribe(self) -> None:
        # self.remove_all_subscriptions()
        pass

    def on_some_event(self, event_data) -> None:
        """Callback invoked when SomeTag event is received."""
        self.agent.machine.pick_job()
```
