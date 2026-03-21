"""Notification data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NotificationIntent(BaseModel):
    """A structured notification request.

    Carries the message content, severity level, and optional metadata so that
    any :class:`~agentspype.runner.notifications.service.NotificationService`
    implementation can route and format it appropriately.
    """

    title: str | None = Field(
        default=None,
        description="Optional short title or subject for the notification.",
    )
    message: str = Field(
        description="The notification body text.",
    )
    level: str = Field(
        default="info",
        description="Severity level: info, warning, error, or critical.",
    )
    source: str | None = Field(
        default=None,
        description="Identifier for the component that emitted the notification.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for extra context.",
    )


class NotificationConfig(BaseModel):
    """Base notification configuration.

    Downstream projects can subclass this to add channel-specific fields
    (e.g. Telegram, Email).
    """

    enabled: bool = Field(
        default=True,
        description="Whether the notification service is active.",
    )
