# Installation

## Requirements

- Python 3.13 or later
- [Graphviz](https://graphviz.org/) (only required if you use the visualization features)

## Installing from PyPI

### pip

```bash
pip install agentspype
```

### uv

```bash
uv add agentspype
```

### poetry

```bash
poetry add agentspype
```

## Dependencies

AgentsPype depends on the following packages, which are installed automatically:

| Package | Version | Purpose |
|---|---|---|
| `pydantic` | >=2.10.4, <3 | Configuration and status models |
| `python-statemachine` | <2.4.0 | Finite state machine base |
| `pydot` | >=3.0.3, <4 | Visualization (graph generation) |
| `bidict` | >=0.23.1, <1 | Bidirectional mapping in Agency |
| `eventspype` | >=1.1.0, <2 | Event pub/sub |

## Visualization Support

The visualization subsystem uses `pydot` (already a dependency) and delegates diagram rendering to Graphviz. If Graphviz is not installed, calls to `agent.visualize()` or other visualization methods will raise an error at the point where a PNG is written.

Install Graphviz:

**macOS (Homebrew):**
```bash
brew install graphviz
```

**Ubuntu / Debian:**
```bash
sudo apt-get install graphviz
```

**Windows:**
Download the installer from [graphviz.org/download](https://graphviz.org/download/).

## Installing for Development

Clone the repository and install all dependency groups with uv:

```bash
git clone https://github.com/gianlucapagliara/agentspype.git
cd agentspype
uv sync --all-groups
```

This installs the project plus the `dev` and `docs` extras:

- `dev`: pytest, mypy, ruff, pre-commit
- `docs`: mkdocs

## Verifying the Installation

```python
import agentspype
from agentspype.agent.agent import Agent
from agentspype.agency import Agency

print("AgentsPype installed successfully")
print(f"Active agents: {Agency.get_active_agents()}")
```
