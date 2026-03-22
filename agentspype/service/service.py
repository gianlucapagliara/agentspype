"""AgentRunnerService — generic processpype Service wrapping an AgentRunner."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from typing import TYPE_CHECKING, Any

from processpype.service.base import Service

from agentspype.runner.runner import AgentRunner

from .configuration import AgentRunnerServiceConfiguration
from .events import QueueEventSubscriber
from .manager import AgentRunnerManager
from .models import (
    AgentDetailModel,
    AgentSummaryModel,
    HealthModel,
    RunnerStatusModel,
)
from .router import AgentRunnerRouter

if TYPE_CHECKING:
    from processpype.service.manager import ServiceManager

__all__ = ["AgentRunnerService"]

logger = logging.getLogger(__name__)


class AgentRunnerService(Service):
    """Generic processpype Service that manages an :class:`AgentRunner`.

    Subclasses **must** implement :meth:`create_runner` to build the runner
    from ``self.config``.  Everything else (lifecycle, REST endpoints, SSE
    streaming) is handled by this base class.
    """

    configuration_class = AgentRunnerServiceConfiguration

    if TYPE_CHECKING:
        config: AgentRunnerServiceConfiguration

    def __init__(self, name: str | None = None) -> None:
        self._runner: AgentRunner | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        # Strong reference so eventspype's weak-ref bookkeeping keeps the bridge alive.
        self._event_subscriber: QueueEventSubscriber = QueueEventSubscriber()
        super().__init__(name=name)

    # ------------------------------------------------------------------
    # Abstract / overridable factory methods
    # ------------------------------------------------------------------

    def create_runner(self) -> AgentRunner:
        """Build and return an :class:`AgentRunner` from ``self.config``.

        Subclasses **must** implement this method.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError(
            "Subclasses must implement create_runner() to build an AgentRunner "
            "from self.config."
        )

    def create_manager(self) -> ServiceManager:
        return AgentRunnerManager(self.logger)

    def create_router(self) -> AgentRunnerRouter:
        return AgentRunnerRouter(
            name=self.name,
            get_status=lambda: self.status,
            start_service=self.start,
            stop_service=self.stop,
            configure_service=self.configure,
            configure_and_start_service=self.configure_and_start,
            get_agents_summary=self.get_agents_summary,
            get_agent_detail=self.get_agent_detail,
            stop_agent=self.stop_agent,
            get_runner_status=self.get_runner_status,
            get_health=self.get_health,
            event_subscriber=self._event_subscriber,
        )

    def requires_configuration(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        runner = self.create_runner()
        self._runner = runner
        await runner.setup()

        self._subscribe_events()

        await super().start()

        if self.config and self.config.close_application_on_end:
            self._monitor_task = asyncio.create_task(self._monitor_agents())

    async def stop(self) -> None:
        await super().stop()

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        if self._runner is not None:
            await self._runner.teardown()
            self._runner = None

    # ------------------------------------------------------------------
    # Data methods (generic — no domain knowledge)
    # ------------------------------------------------------------------

    def get_agents_summary(self) -> list[AgentSummaryModel]:
        """Return summary of all agents."""
        if self._runner is None:
            return []
        return [
            AgentSummaryModel(
                index=i,
                name=agent.__class__.__name__,
                complete_name=agent.complete_name,
                is_initial=agent.is_initial,
                is_final=agent.is_final,
                state=str(agent.machine.current_state),
            )
            for i, agent in enumerate(self._runner.agents)
        ]

    def get_agent_detail(self, index: int) -> AgentDetailModel:
        """Return detailed info for agent at *index*."""
        if self._runner is None:
            raise IndexError("Runner is not running")
        agents = self._runner.agents
        if index < 0 or index >= len(agents):
            raise IndexError(f"Agent index {index} out of range")
        agent = agents[index]
        config = agent.configuration
        return AgentDetailModel(
            index=index,
            name=agent.__class__.__name__,
            complete_name=agent.complete_name,
            is_initial=agent.is_initial,
            is_final=agent.is_final,
            state=str(agent.machine.current_state),
            status=agent.status.model_dump(),
            configuration=config.model_dump()
            if hasattr(config, "model_dump")
            else dict(vars(config)),
        )

    async def stop_agent(self, index: int) -> dict[str, Any]:
        """Stop agent at *index*."""
        if self._runner is None:
            raise IndexError("Runner is not running")
        agents = self._runner.agents
        if index < 0 or index >= len(agents):
            raise IndexError(f"Agent index {index} out of range")
        agents[index].machine.safe_stop()
        return {"status": "stopped", "index": index}

    def get_runner_status(self) -> RunnerStatusModel:
        """Return runner metadata."""
        now = time.time()
        if self._runner is None:
            return RunnerStatusModel(
                start_time=0.0,
                uptime_seconds=0.0,
                agent_count=0,
                keep_running=False,
            )
        start_time = self._runner.start_time
        return RunnerStatusModel(
            start_time=start_time,
            uptime_seconds=now - start_time if start_time else 0.0,
            agent_count=len(self._runner.agents),
            keep_running=self._runner.keep_running,
        )

    def get_health(self) -> HealthModel:
        """Return health check."""
        return HealthModel(
            status="ok" if self._runner is not None else "error",
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Event bridge wiring
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        """Register the event subscriber on all agent publications."""
        if self._runner is None:
            return
        for agent in self._runner.agents:
            publishing = agent.publishing
            for publication in publishing.get_event_definitions().values():
                publishing.add_subscriber(publication, self._event_subscriber)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _monitor_agents(self) -> None:
        """Poll until all agents reach final state, then send SIGTERM."""
        poll_interval = 1.0
        while True:
            if self._runner is not None and self._runner.agents:
                all_final = all(
                    agent.machine.current_state.final for agent in self._runner.agents
                )
                if all_final:
                    break
            await asyncio.sleep(poll_interval)

        logger.info("All agents reached final state — closing application...")
        os.kill(os.getpid(), signal.SIGTERM)
