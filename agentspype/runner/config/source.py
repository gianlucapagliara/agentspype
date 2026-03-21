"""Config source protocol and result dataclass.

Defines the :class:`ConfigSource` protocol that abstracts where agent YAML
configuration comes from (local file, cloud storage, etc.), the
:class:`ConfigLoadResult` dataclass returned by every source, and the
canonical :func:`compute_config_hash` hashing utility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash for a config dict.

    Keys are sorted recursively via ``json.dumps(sort_keys=True)`` so that
    insertion order does not affect the result.
    """
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class ConfigLoadResult:
    """Result of loading agent configuration from a config source.

    When ``config_hash`` is left empty it is automatically computed from
    ``config`` via :func:`compute_config_hash`.
    """

    config: dict[str, Any]
    config_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = compute_config_hash(self.config)


@runtime_checkable
class ConfigSource(Protocol):
    """Protocol for loading agent YAML configuration from any backend.

    Implementors must provide a single ``load()`` coroutine that returns a
    :class:`ConfigLoadResult`.
    """

    async def load(self) -> ConfigLoadResult:
        """Load the agent configuration and return the result."""
        ...
