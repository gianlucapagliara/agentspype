# Contributing

Thank you for your interest in contributing to AgentsPype. This document covers how to set up a development environment, run tests, check code quality, and submit changes.

## Development Environment

### Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Graphviz](https://graphviz.org/) (for visualization tests)

### Setup

```bash
git clone https://github.com/gianlucapagliara/agentspype.git
cd agentspype

# Install all dependencies including dev and docs extras
uv sync --all-groups
```

This installs:

- **Runtime dependencies:** pydantic, python-statemachine, pydot, bidict, eventspype
- **Dev dependencies:** pytest, pytest-cov, pytest-asyncio, pytest-timeout, mypy, ruff, pre-commit
- **Docs dependencies:** mkdocs

### Pre-commit Hooks

```bash
uv run pre-commit install
```

Pre-commit runs ruff (lint + format) and mypy on every commit.

## Running Tests

```bash
uv run pytest
```

### Coverage Report

```bash
uv run pytest --cov=agentspype --cov-report=term-missing
```

### Specific Tests

```bash
uv run pytest tests/test_agent.py
uv run pytest tests/test_agency.py
uv run pytest -k "test_state_machine_transitions"
```

### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
timeout = 30
testpaths = ["tests"]
```

## Code Quality

### Linting and Formatting (ruff)

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix fixable errors
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking (mypy)

```bash
uv run mypy agentspype
```

Mypy is configured in strict mode in `pyproject.toml`. All public functions and methods must have type annotations. The `ignore_missing_imports = true` option is set to handle third-party packages without stubs.

## Project Structure

```
agentspype/
├── agentspype/
│   ├── __init__.py
│   ├── agency.py                    # Agency registry
│   └── agent/
│       ├── __init__.py
│       ├── agent.py                 # Agent base class
│       ├── configuration.py         # AgentConfiguration
│       ├── definition.py            # AgentDefinition
│       ├── listening.py             # AgentListening
│       ├── publishing.py            # AgentPublishing, StateAgentPublishing
│       ├── state_machine.py         # AgentStateMachine, BasicAgentStateMachine
│       └── status.py                # AgentStatus
│   └── visualization/
│       ├── __init__.py
│       ├── agent_visualization.py
│       ├── base_visualization.py
│       ├── listening_visualization.py
│       ├── publishing_visualization.py
│       └── state_machine_visualization.py
├── tests/
│   ├── test_agent.py
│   └── test_agency.py
├── docs/                            # MkDocs documentation
├── example.py                       # Usage example
├── mkdocs.yml
└── pyproject.toml
```

## Writing Tests

Tests live in the `tests/` directory and use pytest. The project uses `pytest-asyncio` in auto mode, so async test functions work without any decorator.

### Test Patterns

**Creating test agents:**

```python
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from statemachine import State


class TestStateMachine(AgentStateMachine):
    starting = State("Starting", initial=True)
    end = State("End", final=True)
    stop = starting.to(end)

    def after_transition(self, event, state):
        pass


class TestListening(AgentListening):
    def subscribe(self): pass
    def unsubscribe(self): pass


class TestAgent(Agent):
    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=StateAgentPublishing,
        events_listening_class=TestListening,
        state_machine_class=TestStateMachine,
        status_class=AgentStatus,
    )
```

**Cleaning up Agency between tests:**

```python
import pytest
from agentspype.agency import Agency

@pytest.fixture(autouse=True)
def clear_agency():
    Agency.initialized_agents.clear()
    yield
    Agency.initialized_agents.clear()
```

**Using fixtures for agents:**

```python
import pytest
from collections.abc import Generator

@pytest.fixture
def test_agent() -> Generator[TestAgent]:
    agent = TestAgent({})
    yield agent
    try:
        agent.teardown()
    except Exception:
        pass
```

## Building Documentation

```bash
# Serve locally with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

The docs are served at `http://127.0.0.1:8000` during `mkdocs serve`.

## Submitting Changes

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature/my-change`
3. Make your changes, ensuring tests pass and type checks are clean.
4. Commit with a descriptive message.
5. Push and open a pull request against `main`.

### Pull Request Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] No mypy errors (`uv run mypy agentspype`)
- [ ] No ruff errors (`uv run ruff check .`)
- [ ] New public APIs have type annotations
- [ ] New functionality has tests

## Reporting Issues

Open an issue at [github.com/gianlucapagliara/agentspype/issues](https://github.com/gianlucapagliara/agentspype/issues). Include:

- Python version and OS
- AgentsPype version (`pip show agentspype`)
- A minimal reproducible example
- The full traceback if applicable
