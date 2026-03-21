"""AgentRunner — base class for agent lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agentspype.agency import Agency
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.runner.history.service import HistoryService, LogHistoryService
from agentspype.runner.history.utils import get_launch_hash
from agentspype.runner.notifications.service import (
    LogNotificationService,
    NotificationService,
)
from agentspype.runner.runtime import BaseRuntime, set_runtime
from agentspype.runner.shutdown import ShutdownHandler

logger = logging.getLogger(__name__)


class AgentRunner:
    """Run a set of agents through their full lifecycle.

    Subclasses can override the ``on_*`` hooks to inject custom behaviour at
    each phase without having to re-implement the core flow.

    Parameters
    ----------
    agent_configs:
        A sequence of ``AgentConfiguration`` instances (or plain dicts that
        will be forwarded to :meth:`Agency.get_agent_from_configuration`).
        Each entry results in one :class:`Agent` being instantiated during
        :meth:`setup`.
    runtime:
        An optional :class:`BaseRuntime` instance. When ``None`` a fresh
        ``BaseRuntime`` is created automatically.
    shutdown_timeout:
        Maximum number of seconds to wait for agents to reach their final
        state during :meth:`teardown`.
    keep_running:
        When ``True`` the runner will not exit even if all agents reach a
        final state; it will keep the event loop alive and wait for an
        explicit shutdown signal (SIGINT/SIGTERM) instead.  This is useful
        in **service mode** where the runner should stay alive to accept
        new work or external triggers even after the initial batch of
        agents has finished processing.
    history_service:
        An optional :class:`HistoryService` implementation for persisting
        configuration and launch history.  When ``None`` a
        :class:`LogHistoryService` is used (log-only, no persistence).
    notification_service:
        An optional :class:`NotificationService` implementation for sending
        notifications.  When ``None`` a :class:`LogNotificationService` is
        used (log-only, no external channels).
    clock_config:
        An optional ``ClockConfig`` instance (from chronopype). When provided
        the runner will create a clock, register it in the runtime under the
        key ``"clock"``, and manage its lifecycle (start in ``run``, stop in
        ``teardown``). Requires chronopype to be installed.
    """

    def __init__(
        self,
        agent_configs: list[AgentConfiguration],
        *,
        runtime: BaseRuntime | None = None,
        shutdown_timeout: float = 10.0,
        keep_running: bool = False,
        history_service: HistoryService | None = None,
        notification_service: NotificationService | None = None,
        clock_config: Any | None = None,
    ) -> None:
        self._agent_configs = list(agent_configs)
        self._runtime = runtime or BaseRuntime()
        self._agents: list[Agent] = []
        self._shutdown_timeout = shutdown_timeout
        self._keep_running = keep_running
        self._history_service: HistoryService = history_service or LogHistoryService()
        self._notification_service: NotificationService = (
            notification_service or LogNotificationService()
        )
        self._start_time: float = 0.0
        self._config_hash: str = ""
        self._launch_hash: str = ""
        self._clock_config = clock_config
        self._clock: Any | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agents(self) -> list[Agent]:
        """Snapshot of instantiated agents (available after :meth:`setup`)."""
        return list(self._agents)

    @property
    def runtime(self) -> BaseRuntime:
        return self._runtime

    @property
    def history_service(self) -> HistoryService:
        """The history service used by this runner."""
        return self._history_service

    @property
    def notification_service(self) -> NotificationService:
        """The notification service used by this runner."""
        return self._notification_service

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    async def on_setup(self) -> None:
        """Called at the beginning of :meth:`setup`, before agents are created."""

    async def on_run_start(self) -> None:
        """Called at the start of :meth:`run`, after setup completes."""

    async def on_run_end(self) -> None:
        """Called when the run loop exits, before teardown begins."""

    async def on_teardown(self) -> None:
        """Called at the end of :meth:`teardown`, after all cleanup."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Initialize runtime, instantiate and start agents."""
        await self.on_setup()

        set_runtime(self._runtime)
        logger.info("Runtime set")

        # Create and register clock if configured
        if self._clock_config is not None:
            self._clock = self._create_clock(self._clock_config)
            self._runtime.register("clock", self._clock)
            logger.info(
                "Clock registered in runtime (mode=%s)", self._clock.clock_mode.name
            )

        # Start notification service
        await self._notification_service.start()

        # Instantiate agents from configurations
        for config in self._agent_configs:
            agent = Agency.get_agent_from_configuration(config)
            self._agents.append(agent)
            logger.info("Instantiated agent: %s", agent.complete_name)

        # Start all agents
        for agent in self._agents:
            agent.machine.safe_start()

        logger.info("Setup complete — %d agent(s) started", len(self._agents))

        # Record history
        if self._history_service.enabled:
            self._start_time = time.time()
            self._history_service.save_config(
                config_hash=self._config_hash,
                config={},
                metadata=None,
            )
            self._launch_hash = get_launch_hash(self._config_hash, self._start_time)
            self._history_service.save_launch_start(
                launch_hash=self._launch_hash,
                config_hash=self._config_hash,
                metadata=None,
            )

    async def run(self) -> None:
        """Setup, run the main loop, and teardown on exit."""
        shutdown = ShutdownHandler()
        shutdown.install()

        try:
            await self.setup()
            await self.on_run_start()

            logger.info(
                "Runner started with %d agent(s) — waiting for shutdown signal",
                len(self._agents),
            )

            if self._clock is not None:
                async with self._clock:
                    # Start the clock's run loop as a background task
                    clock_task = asyncio.create_task(self._clock.run())
                    try:
                        await self._run_loop(shutdown)
                    finally:
                        # Shutdown clock before exiting context
                        await self._clock.shutdown()
                        try:
                            await clock_task
                        except (asyncio.CancelledError, Exception):
                            pass
            else:
                await self._run_loop(shutdown)

            await self.on_run_end()
        except Exception:
            logger.exception("Runner encountered an error")
            raise
        finally:
            await self.teardown()

    async def teardown(self) -> None:
        """Stop agents gracefully, clean up runtime resources."""
        # Stop all agents
        for agent in self._agents:
            try:
                agent.machine.safe_stop()
            except Exception:
                logger.warning(
                    "Error stopping agent %s", agent.complete_name, exc_info=True
                )

        # Wait for agents to reach final state
        await self._wait_agents_final(self._shutdown_timeout)

        # Clean up runtime
        try:
            await self._runtime.teardown()
        except Exception:
            logger.warning("Error during runtime teardown", exc_info=True)
        finally:
            set_runtime(None)

        # Stop notification service
        try:
            await self._notification_service.stop()
        except Exception:
            logger.warning("Error stopping notification service", exc_info=True)

        # Record launch end in history
        if self._history_service.enabled and self._launch_hash:
            self._history_service.save_launch_end(
                launch_hash=self._launch_hash,
                metadata=None,
            )

        self._agents.clear()
        await self.on_teardown()
        logger.info("Runner stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_loop(self, shutdown: ShutdownHandler) -> None:
        """Wait for shutdown signal or all agents reaching final state."""
        poll_interval = 1.0

        while not shutdown.is_set:
            if not self._keep_running and self._agents:
                all_final = all(
                    agent.machine.current_state.final for agent in self._agents
                )
                if all_final:
                    logger.info("All agents reached final state — shutting down")
                    break
            await asyncio.sleep(poll_interval)

    async def _wait_agents_final(self, timeout: float) -> None:
        """Poll until all agents reach a final state or *timeout* expires."""
        poll_interval = 0.25
        elapsed = 0.0

        while elapsed < timeout:
            if all(agent.machine.current_state.final for agent in self._agents):
                return
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        stuck = [
            agent.complete_name
            for agent in self._agents
            if not agent.machine.current_state.final
        ]
        if stuck:
            logger.warning(
                "Shutdown timeout (%.1fs) reached. %d agent(s) not final: %s",
                timeout,
                len(stuck),
                ", ".join(stuck),
            )

    @staticmethod
    def _create_clock(clock_config: Any) -> Any:
        """Create a clock instance from a ClockConfig.

        Imports chronopype lazily so the runner module can be imported without
        chronopype installed.
        """
        try:
            from chronopype import BacktestClock, ClockMode, RealtimeClock
        except ImportError as e:
            raise ImportError(
                "chronopype is required for clock support. "
                "Install it with: pip install agentspype[clock]"
            ) from e

        if clock_config.clock_mode is ClockMode.BACKTEST:
            return BacktestClock(clock_config)
        return RealtimeClock(clock_config)
