"""AgentRunner — base class for agent lifecycle management."""

from __future__ import annotations

import asyncio
import logging

from agentspype.agency import Agency
from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
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
    """

    def __init__(
        self,
        agent_configs: list[AgentConfiguration],
        *,
        runtime: BaseRuntime | None = None,
        shutdown_timeout: float = 10.0,
        keep_running: bool = False,
    ) -> None:
        self._agent_configs = list(agent_configs)
        self._runtime = runtime or BaseRuntime()
        self._agents: list[Agent] = []
        self._shutdown_timeout = shutdown_timeout
        self._keep_running = keep_running

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

        # Instantiate agents from configurations
        for config in self._agent_configs:
            agent = Agency.get_agent_from_configuration(config)
            self._agents.append(agent)
            logger.info("Instantiated agent: %s", agent.complete_name)

        # Start all agents
        for agent in self._agents:
            agent.machine.safe_start()

        logger.info("Setup complete — %d agent(s) started", len(self._agents))

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
