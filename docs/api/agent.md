# Agent

**Module:** `agentspype.agent.agent`

## `Agent`

Base class for all agents. Owns and wires together the state machine, event listener, event publisher, configuration, and status. Registers itself with `Agency` on construction and deregisters on teardown.

### Class Variables

| Name | Type | Description |
|---|---|---|
| `definition` | `AgentDefinition` | **Required.** Immutable binding of all five component classes. Must be declared on every concrete subclass. |
| `_logger` | `logging.Logger | None` | Lazily initialized per class. Access via `cls.logger()`. |

### Constructor

```python
Agent(
    configuration: AgentConfiguration | dict[str, Any],
    parent_id: int | None = None,
)
```

**Parameters:**

- `configuration` — Either an `AgentConfiguration` instance or a plain dict. If a dict is provided, it is passed to `definition.configuration_class(**configuration)`.
- `parent_id` — Optional Python `id()` of a parent agent. Used to populate `complete_name`. Once set to a non-`None` value it cannot be changed.

**Side effects on construction:**
1. Builds publishing, listening, state machine, and status instances from the definition.
2. Calls `Agency.register_agent(self)`.
3. Calls `self.initialize()`.

### Methods

#### `initialize() -> None`

Called at the end of `__init__`. Override this to run setup logic after all components are ready. Default is a no-op.

```python
def initialize(self) -> None:
    super().initialize()
    self.logger().info("Agent ready")
```

#### `teardown() -> None`

Unsubscribes all event listeners and deregisters the agent from `Agency`. Safe to call multiple times.

Called automatically when:
- The state machine enters a final state (`on_enter_end` hook).
- The agent is garbage-collected (`__del__`, exceptions suppressed).

#### `clone() -> Agent`

Returns a new instance of the same class initialized with the current configuration. The clone goes through the full initialization sequence independently.

```python
copy = agent.clone()
assert type(copy) is type(agent)
assert copy is not agent
```

#### `logger() -> logging.Logger` *(classmethod)*

Returns the class-level logger, creating it lazily on first call. The logger name is the class name.

```python
MyAgent.logger().info("class-level log")
agent.logger().debug("instance log — same logger object")
```

### Properties

#### `name -> str`

Unique identifier string in the format `AID:<id>|<ClassName>` where `<id>` is `id(self)`.

#### `complete_name -> str`

Extended name that includes the parent id: `PID:<parent_id>|AID:<id>|<ClassName>`. If no parent, uses `PID:-`.

#### `configuration -> AgentConfiguration`

The agent's configuration instance.

#### `machine -> AgentStateMachine`

The agent's state machine instance.

#### `listening -> AgentListening`

The agent's event listening instance.

#### `publishing -> AgentPublishing`

The agent's event publishing instance.

#### `status -> AgentStatus`

The agent's runtime status instance.

#### `parent_id -> int | None`

The Python `id()` of the parent agent, or `None`. Can be set once to a non-`None` value; attempting to change it to another non-`None` value raises `ValueError`.

### Visualization Methods

All visualization methods are lazy: the `AgentVisualization` class is imported only when first called. If `pydot` is not installed, an `ImportError` is raised.

#### `visualize(...) -> pydot.Dot`

```python
agent.visualize(
    save_file: bool = False,
    filename: str | None = None,
    output_dir: str = ".diagrams",
    include_state_machine: bool = True,
    include_publishing: bool = True,
    include_listening: bool = True,
    show_current_state: bool = True,
    **kwargs,
) -> pydot.Dot
```

Creates a comprehensive diagram with the agent node at center, optionally including state machine, publishing, and listening subgraphs.

#### `visualize_state_machine(...) -> pydot.Dot`

```python
agent.visualize_state_machine(
    save_file: bool = False,
    filename: str | None = None,
    output_dir: str = ".diagrams",
    **kwargs,
) -> pydot.Dot
```

Visualizes only the state machine component with current state highlighted.

#### `visualize_publishing(...) -> pydot.Dot`

```python
agent.visualize_publishing(
    save_file: bool = False,
    filename: str | None = None,
    output_dir: str = ".diagrams",
    **kwargs,
) -> pydot.Dot
```

Visualizes only the publishing component.

#### `visualize_listening(...) -> pydot.Dot`

```python
agent.visualize_listening(
    save_file: bool = False,
    filename: str | None = None,
    output_dir: str = ".diagrams",
    **kwargs,
) -> pydot.Dot
```

Visualizes only the listening component.

#### `create_all_diagrams(...) -> dict[str, pydot.Dot]`

```python
agent.create_all_diagrams(
    save_files: bool = True,
    output_dir: str = ".diagrams",
    **kwargs,
) -> dict[str, pydot.Dot]
```

Creates and returns all four diagrams keyed by `"state_machine"`, `"publishing"`, `"listening"`, and `"comprehensive"`.
