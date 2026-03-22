"""Service configuration for the agent runner service."""

from __future__ import annotations

from processpype.core.configuration.models import ServiceConfiguration
from pydantic import Field


class AgentRunnerServiceConfiguration(ServiceConfiguration):
    """Configuration for AgentRunnerService.

    Extends processpype's ServiceConfiguration with agent-runner-specific
    options.  Uses ``extra="allow"`` so subclasses in downstream projects
    can add domain-specific fields without redefining model_config.
    """

    model_config = {"extra": "allow", "frozen": False}

    close_application_on_end: bool = Field(
        default=False,
        description="Send SIGTERM when all agents reach final state.",
    )
