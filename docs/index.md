# AgentsPype

**AgentsPype** is a Python framework for building event-driven agents with finite state machines. It provides a structured way to define agents that react to events, transition between states, and publish notifications to other components.

## What is AgentsPype?

AgentsPype combines two complementary concepts:

- **Finite State Machines (FSMs)** — Each agent has an embedded state machine that governs its lifecycle and behavior. States and transitions are declared declaratively using the built-in `agentspype.fsm` engine.
- **Event-driven communication** — Agents subscribe to and publish typed events through [eventspype](https://github.com/gianlucapagliara/eventspype), a pub/sub library. Agents are decoupled: publishers and subscribers never reference each other directly.

The combination makes it easy to build systems where agents respond to external signals, change their internal state accordingly, and notify the rest of the system about what happened.

## Key Features

- **Declarative state machines** — Define states and transitions as class attributes. The `StateMachineMeta` metaclass injects sensible defaults (`starting`, `idle`, `end`) so every agent has a consistent lifecycle out of the box.
- **Typed event pub/sub** — `AgentListening` and `AgentPublishing` wrap eventspype's `MultiSubscriber` and `MultiPublisher`. Events carry structured data classes and are identified by enum tags for type safety.
- **State transition events** — `StateAgentPublishing` publishes a `StateMachineEvent` whenever the state machine transitions, allowing other agents to react to state changes without polling.
- **Central registry** — `Agency` is a class-level registry that tracks all live agents and provides a factory method to instantiate agents from their configuration objects.
- **Pydantic configuration and status** — `AgentConfiguration` and `AgentStatus` are Pydantic `BaseModel` subclasses. Agents can be initialized from plain dicts as well as model instances.
- **Immutable definitions** — `AgentDefinition` is a frozen Pydantic model that binds together the state machine class, listening class, publishing class, configuration class, and status class for a given agent type.
- **Visualization** — Built-in pydot-based visualization for state machines, event publishing, event listening, and combined agent diagrams. Diagrams are saved as PNG files.
- **Agent Runner & CLI** — Orchestrate multi-agent setups from YAML configuration with the `agentspype` CLI (`run`, `create`, `plot` commands) or programmatically via `AgentRunner`.
- **Configuration management** — YAML-based configuration loading with Pydantic validation and an optional interactive wizard (`pydantic-wizard`).
- **Clock support** — Optional time management via [chronopype](https://github.com/gianlucapagliara/chronopype) for real-time and backtesting scenarios (`pip install agentspype[clock]`).
- **Service mode** — Run agents as background services with REST API integration via [processpype](https://github.com/gianlucapagliara/processpype) (`pip install agentspype[service]`).

## Quick Install

```bash
pip install agentspype
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add agentspype
```

## Minimal Example

```python
from agentspype.fsm import State
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus


class MyStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    running = State("Running")
    end = State("End", final=True)

    start = starting.to(running)
    stop = running.to(end)

    def after_transition(self, event, state):
        self.agent.publishing.publish_transition(event, state)


class MyListening(AgentListening):
    def subscribe(self):
        pass  # register event subscriptions here

    def unsubscribe(self):
        pass  # remove event subscriptions here


class MyAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=StateAgentPublishing,
        events_listening_class=MyListening,
        state_machine_class=MyStateMachine,
        status_class=AgentStatus,
    )


agent = MyAgent({})
agent.machine.start()
agent.machine.stop()
```

## Navigation

- **[Getting Started](getting-started/installation.md)** — Install the library and run your first agent.
- **[User Guide](guide/agents.md)** — In-depth explanations of agents, state machines, events, the agency registry, configuration, and visualization.
- **[API Reference](api/agent.md)** — Detailed class and method documentation.
- **[Contributing](development/contributing.md)** — How to set up a development environment and contribute.
