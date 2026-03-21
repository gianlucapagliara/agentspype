"""Notification service Protocol and log-only default implementation.

The :class:`NotificationService` Protocol defines the interface that all
notification implementations must satisfy.  It mirrors the :class:`ConfigSource`
protocol pattern: core owns the protocol and a trivial default.

:class:`LogNotificationService` is the built-in default: it satisfies the
protocol using only Python logging, with no external channel dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from agentspype.runner.notifications.models import (
    NotificationConfig,
    NotificationIntent,
)


def format_notification_message(
    message: str,
    level: str,
    source: str | None,
    metadata: dict[str, Any],
) -> str:
    """Format a notification message with level prefix and optional metadata.

    Shared by all :class:`NotificationService` implementations.
    """
    parts = [f"[{level.upper()}]"]
    if source:
        parts.append(f"[{source}]")
    prefix = "".join(parts)
    if metadata:
        return f"{prefix} {message}\nmetadata={metadata}"
    return f"{prefix} {message}"


@runtime_checkable
class NotificationService(Protocol):
    """Protocol for sending notifications through any channel backend.

    Implementors must provide the listed methods so that the runner and
    application can start/stop the service and route messages without
    knowing the concrete channel details.
    """

    @property
    def enabled(self) -> bool:
        """Return whether notifications are active."""
        ...

    async def start(self) -> None:
        """Start the notification service and connect to channels."""
        ...

    async def stop(self) -> None:
        """Stop the notification service and disconnect from channels."""
        ...

    def notify(
        self,
        message: str,
        level: str = "info",
        **kwargs: Any,
    ) -> None:
        """Send a notification message."""
        ...

    def notify_intent(self, intent: NotificationIntent) -> None:
        """Send a notification from a :class:`NotificationIntent`."""
        ...


class LogNotificationService:
    """Default notification service that routes all messages to Python logging.

    No external channel dependencies.  Satisfies the :class:`NotificationService`
    protocol and serves as the built-in fallback when no concrete implementation
    is provided.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._config = NotificationConfig()

    def configure(self, config: NotificationConfig) -> None:
        """Apply a :class:`NotificationConfig` to control behaviour."""
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    async def start(self) -> None:
        if not self.enabled:
            self._logger.info("Notifications disabled")

    async def stop(self) -> None:
        pass

    def notify(
        self,
        message: str,
        level: str = "info",
        **kwargs: Any,
    ) -> None:
        source = kwargs.get("source")
        metadata = kwargs.get("metadata") or {}
        formatted = format_notification_message(
            message, level=level, source=source, metadata=metadata
        )
        self._logger.info(formatted)

    def notify_intent(self, intent: NotificationIntent) -> None:
        self.notify(
            message=intent.message,
            level=intent.level,
            source=intent.source,
            metadata=intent.metadata,
        )
