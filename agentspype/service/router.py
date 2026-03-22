"""AgentRunnerRouter — generic FastAPI router for the agent runner service."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from processpype.server.service_router import ServiceRouter
from processpype.service.models import ServiceStatus

from .events import QueueEventSubscriber
from .models import (
    AgentDetailModel,
    AgentSummaryModel,
    HealthModel,
    RunnerStatusModel,
)

__all__ = ["AgentRunnerRouter"]

logger = logging.getLogger(__name__)


class AgentRunnerRouter(ServiceRouter):
    """Generic router for agent runner services.

    Provides endpoints for agent management, runner status, and SSE event
    streaming.  Domain-specific subclasses can override
    :meth:`register_custom_routes` to add extra endpoints (e.g. exchanges,
    accounts).

    Endpoints:

    - GET  /agents              — list all agents (summary)
    - GET  /agents/{index}      — single agent detail
    - POST /agents/{index}/stop — stop a specific agent
    - GET  /runner/status       — runner metadata
    - GET  /runner/health       — health check
    - GET  /events              — SSE event stream
    """

    def __init__(
        self,
        name: str,
        get_status: Callable[[], ServiceStatus],
        start_service: Callable[[], Any] | None = None,
        stop_service: Callable[[], Any] | None = None,
        configure_service: Callable[[dict[str, Any]], Any] | None = None,
        configure_and_start_service: Callable[[dict[str, Any]], Any] | None = None,
        get_agents_summary: Callable[[], list[AgentSummaryModel]] | None = None,
        get_agent_detail: Callable[[int], AgentDetailModel] | None = None,
        stop_agent: Callable[[int], Any] | None = None,
        get_runner_status: Callable[[], RunnerStatusModel] | None = None,
        get_health: Callable[[], HealthModel] | None = None,
        event_subscriber: QueueEventSubscriber | None = None,
    ) -> None:
        super().__init__(
            name=name,
            get_status=get_status,
            start_service=start_service,
            stop_service=stop_service,
            configure_service=configure_service,
            configure_and_start_service=configure_and_start_service,
        )

        self._get_agents_summary = get_agents_summary
        self._get_agent_detail = get_agent_detail
        self._stop_agent = stop_agent
        self._get_runner_status = get_runner_status
        self._get_health = get_health
        self._event_subscriber = event_subscriber

        self._setup_agent_routes()
        self._setup_runner_routes()
        self._setup_event_routes()
        self.register_custom_routes()

    # ------------------------------------------------------------------
    # Agent endpoints
    # ------------------------------------------------------------------

    def _setup_agent_routes(self) -> None:
        """Register agent management endpoints."""

        @self.get("/agents")
        async def get_agents() -> list[AgentSummaryModel]:
            """List all agents."""
            if self._get_agents_summary is None:
                raise HTTPException(
                    status_code=501, detail="Agent summary not implemented"
                )
            return self._get_agents_summary()

        @self.get("/agents/{index}")
        async def get_agent(index: int) -> AgentDetailModel:
            """Get detailed info for a single agent."""
            if self._get_agent_detail is None:
                raise HTTPException(
                    status_code=501, detail="Agent detail not implemented"
                )
            try:
                return self._get_agent_detail(index)
            except IndexError as e:
                raise HTTPException(
                    status_code=404, detail=f"Agent {index} not found"
                ) from e

        @self.post("/agents/{index}/stop")
        async def stop_agent(index: int) -> dict[str, Any]:
            """Stop a specific agent."""
            if self._stop_agent is None:
                raise HTTPException(
                    status_code=501, detail="Agent stop not implemented"
                )
            try:
                return await self._stop_agent(index)
            except IndexError as e:
                raise HTTPException(
                    status_code=404, detail=f"Agent {index} not found"
                ) from e

    # ------------------------------------------------------------------
    # Runner endpoints
    # ------------------------------------------------------------------

    def _setup_runner_routes(self) -> None:
        """Register runner status and health endpoints."""

        @self.get("/runner/status")
        async def get_runner_status() -> RunnerStatusModel:
            """Get runner metadata."""
            if self._get_runner_status is None:
                raise HTTPException(
                    status_code=501, detail="Runner status not implemented"
                )
            return self._get_runner_status()

        @self.get("/runner/health")
        async def get_health() -> HealthModel:
            """Health check."""
            if self._get_health is None:
                raise HTTPException(
                    status_code=501, detail="Health check not implemented"
                )
            return self._get_health()

    # ------------------------------------------------------------------
    # SSE event stream
    # ------------------------------------------------------------------

    def _setup_event_routes(self) -> None:
        """Register the SSE event streaming endpoint."""

        @self.get("/events")
        async def get_events() -> StreamingResponse:
            """Server-Sent Events stream of agent lifecycle events."""
            if self._event_subscriber is None:
                raise HTTPException(
                    status_code=501, detail="Event streaming not available"
                )

            return StreamingResponse(
                _sse_generator(self._event_subscriber),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    # ------------------------------------------------------------------
    # Extension hook
    # ------------------------------------------------------------------

    def register_custom_routes(self) -> None:
        """Override in subclasses to add domain-specific routes."""


async def _sse_generator(bridge: QueueEventSubscriber) -> AsyncGenerator[str]:
    """Yield SSE-formatted events from the event bridge queue.

    Sends a keepalive comment every 30 seconds to prevent proxy/client
    timeouts.  Cleans up the queue subscription on client disconnect.
    """
    queue = bridge.subscribe_consumer()
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                event_type = event.get("event_type", "message")
                data = json.dumps(event, default=str)
                yield f"event: {event_type}\ndata: {data}\n\n"
            except TimeoutError:
                yield ": keepalive\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        bridge.unsubscribe_consumer(queue)
