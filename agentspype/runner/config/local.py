"""Local YAML file configuration source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentspype.runner.config.source import ConfigLoadResult, compute_config_hash


class LocalConfigSource:
    """Load agent configuration from a local YAML file."""

    def __init__(self, config_file: Path) -> None:
        self._config_file = config_file

    async def load(self) -> ConfigLoadResult:
        """Read the YAML file and return a :class:`ConfigLoadResult`."""
        config = self._read_yaml()
        return ConfigLoadResult(
            config=config,
            config_hash=f"LOCAL_{compute_config_hash(config)}",
        )

    def _read_yaml(self) -> dict[str, Any]:
        if not self._config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_file}"
            )
        with open(self._config_file) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        return data
