# Visualization

AgentsPype includes a visualization subsystem that generates pydot diagrams for agents, their state machines, event publishers, and event listeners. Diagrams can be rendered to PNG files using Graphviz.

## Prerequisites

Visualization requires Graphviz to be installed on the system. The `pydot` Python package (already a dependency) generates DOT language descriptions; Graphviz renders them to images.

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

## Agent-Level Visualization

Every `Agent` instance provides visualization methods directly:

### `agent.visualize()`

Creates a comprehensive diagram combining the state machine, publisher, and listener into a single graph. The central agent node (pink) connects to the state machine (via a red edge), the publisher (green edge), and the listener (blue edge).

```python
graph = agent.visualize(
    save_file=True,                     # write to disk
    filename="my_agent",                # base name (no extension)
    output_dir=".diagrams",             # directory to write into
    include_state_machine=True,         # include FSM subgraph
    include_publishing=True,            # include publishing subgraph
    include_listening=True,             # include listening subgraph
    show_current_state=True,            # highlight active state in gold
)
```

Returns a `pydot.Dot` object. If `save_file=True`, saves `<output_dir>/<filename>.png`.

### `agent.visualize_state_machine()`

Creates a diagram of the state machine only, using python-statemachine's built-in `DotGraphMachine` with custom AgentsPype styling applied on top.

```python
graph = agent.visualize_state_machine(
    save_file=True,
    filename="my_agent_fsm",
    output_dir=".diagrams",
)
```

State node colors:
- **Light blue** — regular state
- **Gold / orange border** — current state (highlighted when `show_current_state=True`)
- **Light green** — initial state marker (`i` node)
- **Light coral** — final state marker

Edge colors:
- **Dark green, dashed** — `start` transition
- **Dark red, dashed** — `stop` transition
- **Dark blue, bold** — other named transitions

### `agent.visualize_publishing()`

Creates a diagram of the agent's event publishing component. Each `EventPublication` declared on the publishing class becomes a node connected to its event tag and event data class.

```python
graph = agent.visualize_publishing(save_file=True)
```

### `agent.visualize_listening()`

Creates a diagram of the agent's event listening component. Each public callable method on the listening class appears as a callback node.

```python
graph = agent.visualize_listening(save_file=True)
```

### `agent.create_all_diagrams()`

Creates all four diagrams (state machine, publishing, listening, comprehensive) and returns them as a dict:

```python
diagrams = agent.create_all_diagrams(save_files=True, output_dir=".diagrams")
# diagrams == {
#     "state_machine": pydot.Dot,
#     "publishing": pydot.Dot,
#     "listening": pydot.Dot,
#     "comprehensive": pydot.Dot,
# }

for name, graph in diagrams.items():
    print(f"{name}: {len(graph.get_node_list())} nodes")
```

## Visualization Classes

The visualization subsystem has four classes, all under `agentspype.visualization`:

### `BaseVisualization`

Abstract base providing shared graph-building helpers:

- `create_graph(name)` — creates a `pydot.Dot` with default styling (Arial font, left-to-right layout)
- `create_node(node_id, label, ...)` — creates a styled `pydot.Node`
- `create_edge(source, target, label, ...)` — creates a styled `pydot.Edge`
- `save_diagram(graph, filename, output_dir)` — writes a PNG to disk
- `visualize(target, save_file, filename, output_dir, ...)` — calls `create_visualization` then optionally saves

All subclasses must implement the abstract `create_visualization(target, graph, **kwargs)` method.

### `StateMachineVisualization`

Wraps python-statemachine's `DotGraphMachine` and applies custom AgentsPype styling. The `visualize_with_current_state(state_machine)` method highlights the currently active state.

### `PublishingVisualization`

Takes an `AgentPublishing` class (or instance) and renders its `EventPublication` attributes as a graph. Calls `get_event_definitions()` to enumerate publications.

### `ListeningVisualization`

Takes an `AgentListening` class (or instance) and renders its callback methods as a graph. Calls `get_event_definitions()` to enumerate subscriptions.

### `AgentVisualization`

The top-level orchestrator. Owns instances of the three component visualizers and merges their graphs into a single comprehensive diagram. Also provides per-component convenience methods (`visualize_state_machine_only`, `visualize_publishing_only`, `visualize_listening_only`, `create_component_diagrams`).

## Direct Use of Visualization Classes

You can use the component visualizers directly without an agent instance:

```python
from agentspype.visualization.state_machine_visualization import StateMachineVisualization
from agentspype.visualization.publishing_visualization import PublishingVisualization
from agentspype.visualization.listening_visualization import ListeningVisualization

# Visualize just the state machine class (no instance needed)
viz = StateMachineVisualization()
graph = viz.visualize(agent.machine, save_file=True, filename="fsm")

# Visualize a publishing class
pub_viz = PublishingVisualization()
graph = pub_viz.visualize(WorkerPublishing, save_file=True, filename="publishing")

# Visualize a listening class
listen_viz = ListeningVisualization()
graph = listen_viz.visualize(WorkerListening, save_file=True, filename="listening")
```

## Working with pydot Graphs

All visualization methods return a `pydot.Dot` object. You can manipulate the graph before saving:

```python
graph = agent.visualize()

# Add a custom note
import pydot
note = pydot.Node("note", label="Generated 2026-01-01", shape="note", fillcolor="lightyellow")
graph.add_node(note)

# Render to different formats
graph.write_svg(".diagrams/agent.svg")
graph.write_pdf(".diagrams/agent.pdf")

# Or get the DOT source
print(graph.to_string())
```

## Example

```python
from agentspype.agent.agent import Agent
from agentspype.agent.definition import AgentDefinition
# ... (define all components as in quickstart) ...

agent = ExampleAgent({})

# Drive the state machine to a mid-run state
agent.machine.begin_processing()
agent.machine.wait_for_input()

# Visualize with current state highlighted
agent.visualize(
    save_file=True,
    filename="example_mid_run",
    output_dir=".diagrams",
    show_current_state=True,  # highlights "Waiting" state in gold
)

# Quick state machine only
agent.visualize_state_machine(save_file=True)

# Clean up
agent.teardown()
```

After running, check the `.diagrams/` directory for the generated PNG files.
