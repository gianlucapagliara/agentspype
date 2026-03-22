"""No-op service manager for the agent runner service.

The AgentRunner handles its own lifecycle; the manager satisfies the
processpype ``ServiceManager`` contract without adding extra logic.
"""

from __future__ import annotations

from processpype.core.service.manager import ServiceManager


class AgentRunnerManager(ServiceManager):
    """No-op manager — lifecycle is handled by AgentRunnerService directly."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass
