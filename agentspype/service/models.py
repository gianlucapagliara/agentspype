"""Pydantic response models for the agent runner service.

These models are pure Pydantic and have no dependency on processpype,
so they can be imported and used even without the ``[service]`` extra.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentSummaryModel(BaseModel):
    """Summary information for a single agent."""

    index: int
    name: str
    complete_name: str
    is_initial: bool
    is_final: bool
    state: str


class AgentDetailModel(AgentSummaryModel):
    """Detailed information for a single agent, including status and config."""

    status: dict[str, Any]
    configuration: dict[str, Any]


class RunnerStatusModel(BaseModel):
    """Runner metadata."""

    start_time: float
    uptime_seconds: float
    agent_count: int
    keep_running: bool


class HealthModel(BaseModel):
    """Health check response."""

    status: str
    timestamp: float
