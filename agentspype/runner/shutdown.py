"""Graceful shutdown handling via SIGINT/SIGTERM."""

from __future__ import annotations

import asyncio
import logging
import os
import signal

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """Installs signal handlers and exposes an asyncio Event for clean shutdown.

    On the first signal the event is set so that coroutines polling
    :attr:`is_set` or awaiting :meth:`wait` can begin graceful cleanup.
    A second signal triggers an immediate ``os._exit(1)`` to force-quit
    when graceful shutdown is stuck.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._event = asyncio.Event()
        self._loop = loop
        self._force_exit_on_next = False

    def install(self) -> None:
        """Register SIGINT and SIGTERM handlers on the running event loop."""
        loop = self._loop or asyncio.get_running_loop()
        self._loop = loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle, sig)

    def _handle(self, sig: signal.Signals) -> None:
        if self._force_exit_on_next:
            logger.warning("Received %s again — forcing immediate exit", sig.name)
            # os._exit is intentional here: on a second signal we need to
            # terminate immediately, bypassing Python cleanup (atexit handlers,
            # finally blocks, etc.) which may be stuck or deadlocked — this is
            # exactly the scenario os._exit is designed for vs sys.exit.
            os._exit(1)

        logger.info("Received %s — initiating graceful shutdown", sig.name)
        self._event.set()
        self._force_exit_on_next = True

    async def wait(self) -> None:
        """Block until a shutdown signal is received."""
        await self._event.wait()

    @property
    def is_set(self) -> bool:
        """Return ``True`` after the first shutdown signal has been received."""
        return self._event.is_set()
