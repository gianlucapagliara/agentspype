"""History service protocol and log-only default implementation.

The :class:`HistoryService` protocol defines the interface for persisting
configuration and launch history.  It follows the same pattern as
:class:`~agentspype.runner.config.source.ConfigSource`: the framework owns the
protocol and a trivial default, while downstream projects can provide full
persistence implementations (e.g. databases, cloud storage).

:class:`LogHistoryService` is the built-in default that simply logs history
events via Python's ``logging`` module without persisting anything.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class HistoryService(Protocol):
    """Protocol for persisting configuration and launch history.

    Implementors are called by the runner after config load
    (:meth:`save_config`) and at run start/end (:meth:`save_launch_start`,
    :meth:`save_launch_end`).
    """

    @property
    def enabled(self) -> bool:
        """Whether history persistence is active."""
        ...

    def save_config(
        self,
        config_hash: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a configuration snapshot."""
        ...

    def save_launch_start(
        self,
        launch_hash: str,
        config_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a run has started."""
        ...

    def save_launch_end(
        self,
        launch_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a run has ended."""
        ...


class LogHistoryService:
    """Default history service that logs events without persisting.

    Satisfies the :class:`HistoryService` protocol.  Use when history
    persistence is disabled or when the caller does not inject a concrete
    implementation.
    """

    @property
    def enabled(self) -> bool:
        """Always returns ``True`` — logging is always available."""
        return True

    def save_config(
        self,
        config_hash: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log the config save event."""
        logger.debug(
            "History: config saved (hash=%s, metadata=%s)",
            config_hash,
            metadata,
        )

    def save_launch_start(
        self,
        launch_hash: str,
        config_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log the launch start event."""
        logger.debug(
            "History: launch started (launch_hash=%s, config_hash=%s, metadata=%s)",
            launch_hash,
            config_hash,
            metadata,
        )

    def save_launch_end(
        self,
        launch_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log the launch ended event."""
        logger.debug(
            "History: launch ended (launch_hash=%s, metadata=%s)",
            launch_hash,
            metadata,
        )
