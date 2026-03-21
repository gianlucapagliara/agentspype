"""Runner configuration model."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunnerConfig(BaseModel):
    """Minimal configuration for running agents from a YAML file.

    Users can subclass this model to add domain-specific fields.
    """

    config_file: Path
    log_level: str = Field(default="INFO")
    shutdown_timeout: float = Field(
        default=300.0,
        description="Seconds to wait for agents to reach final state before releasing runtime resources.",
    )
    instance_prefix: str | None = Field(
        default=None,
        description="Override for the auto-generated $INSTANCE_PREFIX.",
    )
