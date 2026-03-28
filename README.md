# AgentsPype

[![CI](https://github.com/gianlucapagliara/agentspype/actions/workflows/ci.yml/badge.svg)](https://github.com/gianlucapagliara/agentspype/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gianlucapagliara/agentspype/branch/main/graph/badge.svg)](https://codecov.io/gh/gianlucapagliara/agentspype)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/agentspype)](https://pypi.org/project/agentspype/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://gianlucapagliara.github.io/agentspype/)

A Python framework for building event-driven agents with finite state machines. AgentsPype combines declarative state machine definitions with typed pub/sub communication, built-in visualization, and a CLI runner for orchestrating agent lifecycles.

## Features

- State Machine Agents: Define agents with finite state machines, automatic state transitions, and lifecycle management
- Event-Driven Communication: Type-safe pub/sub via EventsPype with declarative listening and publishing
- Agent Runner: CLI and programmatic runner for orchestrating multi-agent setups from YAML configuration
- Visualization: Generate diagrams for state machines, event publishing, event listening, and full agent architecture
- Configuration Management: YAML-based configuration with Pydantic validation and optional interactive wizard
- Agent Registry: Central `Agency` for tracking active agents and resolving agent classes by name
- Clock Support: Optional time management via Chronopype for real-time and backtesting scenarios
- Service Mode: Run agents as background services with REST API integration via ProcessPype
- Type Safe: Fully typed with MyPy strict mode
- Well Tested: Comprehensive test suite with coverage enforcement

## Installation

```bash
# Using pip
pip install agentspype

# Using uv
uv add agentspype
```

Optional extras:

```bash
# Clock support
uv add agentspype[clock]

# Interactive configuration wizard
uv add agentspype[wizard]

# Service mode
uv add agentspype[service]

# All extras
uv add agentspype[all]
```

## Quick Start

```python
from agentspype.agent import Agent, AgentDefinition
from agentspype.agent.state_machine import BasicAgentStateMachine
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import AgentPublishing
from agentspype.agent.status import AgentStatus


class MyAgent(Agent):
    definition = AgentDefinition(
        state_machine_class=BasicAgentStateMachine,
        configuration_class=AgentConfiguration,
        listening_class=AgentListening,
        publishing_class=AgentPublishing,
        status_class=AgentStatus,
    )

agent = MyAgent(AgentConfiguration(name="my-agent"))
agent.start()
print(agent.state_machine.current_state)
agent.stop()
```

## Core Components

- **Agent**: Base class with lifecycle management, configuration, state machine, and event wiring
  - `AgentDefinition`: Immutable binding of all agent component classes
  - `AgentConfiguration`: Pydantic model for agent settings
  - `AgentStatus`: Runtime status model

- **State Machine**: Custom lightweight finite state machine engine
  - `AgentStateMachine`: Base class with auto-created `starting`, `idle`, `end` states
  - `BasicAgentStateMachine`: Ready-to-use default implementation

- **Events**: Typed pub/sub communication via EventsPype
  - `AgentListening`: Declarative event subscriptions (extends `MultiSubscriber`)
  - `AgentPublishing`: Event publication (extends `MultiPublisher`)
  - `StateAgentPublishing`: Automatically publishes state transition events

- **Runner**: Agent orchestration and CLI
  - `AgentRunner`: Programmatic lifecycle management (setup, run, teardown)
  - `agentspype` CLI: `run`, `create`, and `plot` commands
  - `BaseRuntime`: Typed resource registry for sharing objects across agents

- **Agency**: Central registry for tracking and resolving agent instances

- **Visualization**: Diagram generation via pydot
  - State machine, publishing, listening, and full agent architecture diagrams

## Documentation

Full documentation is available at [gianlucapagliara.github.io/agentspype](https://gianlucapagliara.github.io/agentspype/).

## Development

AgentsPype uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging:

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run type checks
uv run mypy agentspype

# Run linting
uv run ruff check .

# Run pre-commit hooks
uv run pre-commit run --all-files
```
