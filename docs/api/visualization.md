# Visualization

**Module:** `agentspype.visualization`

The visualization subsystem generates pydot diagrams for agents and their components. All classes inherit from `BaseVisualization` and produce `pydot.Dot` objects that can be rendered to PNG (or other Graphviz output formats).

---

## `BaseVisualization`

**Module:** `agentspype.visualization.base_visualization`

Abstract base class for all visualization components. Provides shared graph-building utilities and the top-level `visualize()` orchestration method.

### Class Variables (Style Constants)

| Name | Value | Description |
|---|---|---|
| `FONT_NAME` | `"Arial"` | Font for all graph text |
| `FONT_SIZE` | `"10pt"` | Font size for all graph text |
| `GRAPH_RANKDIR` | `"LR"` | Graph direction (left to right) |

### Class Methods

#### `create_graph(name, graph_type="digraph") -> pydot.Dot`

Creates a `pydot.Dot` graph with default styling applied.

```python
graph = BaseVisualization.create_graph("My Agent")
```

#### `create_node(node_id, label, shape, style, final, fillcolor, color) -> pydot.Node`

Creates a styled `pydot.Node`.

| Parameter | Default | Description |
|---|---|---|
| `node_id` | — | Unique node identifier |
| `label` | — | Display text |
| `shape` | `"rectangle"` | Node shape |
| `style` | `"rounded, filled"` | Node style |
| `final` | `False` | If `True`, draws double border (`peripheries=2`) |
| `fillcolor` | `"white"` | Fill color |
| `color` | `"black"` | Border color |

#### `create_edge(source, target, label, style, color) -> pydot.Edge`

Creates a styled `pydot.Edge`.

#### `save_diagram(graph, filename, output_dir) -> str`

Saves `graph` as a PNG file. Creates `output_dir` if it does not exist. Appends `.png` if not already present. Returns the full file path.

### Methods

#### `create_visualization(target, graph, **kwargs) -> pydot.Dot` *(abstract)*

Implemented by each subclass to build the diagram for `target`. If `graph` is provided, nodes and edges are added to it; otherwise a new graph is created.

#### `visualize(target, save_file, filename, output_dir, **kwargs) -> pydot.Dot`

Calls `create_visualization(target, **kwargs)` then optionally saves the result. Returns the `pydot.Dot` object.

```python
graph = viz.visualize(target, save_file=True, filename="output", output_dir=".diagrams")
```

---

## `StateMachineVisualization`

**Module:** `agentspype.visualization.state_machine_visualization`

Visualizes an `AgentStateMachine` instance using pydot with custom AgentsPype styling.

### Methods

#### `create_visualization(target, graph, current_state, **kwargs) -> pydot.Dot`

| Parameter | Description |
|---|---|
| `target` | An `AgentStateMachine` instance |
| `graph` | Optional existing graph to add into |
| `current_state` | Optional `State` to highlight in gold |

Node styling:
- States in `states_map` → light blue, dark blue border, rounded
- Node `"i"` (initial marker) → light green, dark green border
- Node `"end"` → light coral, dark red border
- Current state → gold fill, orange border, bold

Edge styling:
- `start` → dark green, dashed
- `stop` → dark red, dashed
- Registered events → dark blue, bold
- Empty label → gray, dotted

#### `visualize_with_current_state(state_machine, save_file, filename, output_dir, **kwargs) -> pydot.Dot`

Convenience method that reads `state_machine.current_state` and passes it to `visualize` as `current_state`.

---

## `PublishingVisualization`

**Module:** `agentspype.visualization.publishing_visualization`

Visualizes an `AgentPublishing` class (or instance). Calls `get_event_definitions()` to enumerate `EventPublication` attributes.

### Graph Structure

- Central publisher node (light green, dark green border, double border)
- For each `EventPublication`:
  - Event node (light blue, dark blue border) — connected from publisher with "publishes" label
  - Event tag node (light yellow, orange border) — connected from event with "tagged as" (dashed)
  - Event class node (light cyan, dark blue border) — connected from event with "data type" (dotted)
- If no events: "No Events" placeholder node (gray, dotted)

### Methods

#### `create_visualization(target, graph, **kwargs) -> pydot.Dot`

| Parameter | Description |
|---|---|
| `target` | An `AgentPublishing` class or instance |
| `graph` | Optional existing graph to add into |

---

## `ListeningVisualization`

**Module:** `agentspype.visualization.listening_visualization`

Visualizes an `AgentListening` class (or instance). Calls `get_event_definitions()` to enumerate callback methods.

### Graph Structure

- Central listener node (light blue, dark blue border, double border)
- For each callback:
  - Callback node (light green, dark green border) — connected from listener with "calls" label
  - If event tag is available: tag node (light yellow, orange) → callback with "triggers" (dashed)
  - If publisher class is available: publisher node (light cyan, dark green) → tag or callback with "publishes"
- If no callbacks: "No Subscriptions" placeholder node (gray, dotted)

### Methods

#### `create_visualization(target, graph, **kwargs) -> pydot.Dot`

| Parameter | Description |
|---|---|
| `target` | An `AgentListening` class or instance |
| `graph` | Optional existing graph to add into |

---

## `AgentVisualization`

**Module:** `agentspype.visualization.agent_visualization`

Top-level orchestrator that combines `StateMachineVisualization`, `PublishingVisualization`, and `ListeningVisualization` into a comprehensive agent diagram. This is the class used by `Agent.visualize()` and related methods.

### Constructor

```python
AgentVisualization()
```

Creates instances of the three component visualizers.

### Methods

#### `create_visualization(target, graph, include_state_machine, include_publishing, include_listening, show_current_state, **kwargs) -> pydot.Dot`

Creates the comprehensive diagram. The agent class name becomes the central node (light pink, dark red border, double border). Component subgraphs are merged into the main graph and connected via colored edges:
- Red edge → state machine
- Green edge → publisher
- Blue edge → listener

#### `visualize_state_machine_only(agent, save_file, filename, output_dir, **kwargs) -> pydot.Dot`

Delegates to `StateMachineVisualization.visualize_with_current_state`.

#### `visualize_publishing_only(agent, save_file, filename, output_dir, **kwargs) -> pydot.Dot`

Delegates to `PublishingVisualization.visualize` with `type(agent.publishing)`.

#### `visualize_listening_only(agent, save_file, filename, output_dir, **kwargs) -> pydot.Dot`

Delegates to `ListeningVisualization.visualize` with `type(agent.listening)`.

#### `create_component_diagrams(agent, save_files, output_dir, **kwargs) -> dict[str, pydot.Dot]`

Creates and returns all four diagrams:

```python
{
    "state_machine": pydot.Dot,
    "publishing": pydot.Dot,
    "listening": pydot.Dot,
    "comprehensive": pydot.Dot,
}
```
