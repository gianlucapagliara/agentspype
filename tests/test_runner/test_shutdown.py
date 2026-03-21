"""Tests for ShutdownHandler."""

from __future__ import annotations

import asyncio
import signal

import pytest

from agentspype.runner.shutdown import ShutdownHandler


@pytest.mark.asyncio
async def test_is_set_initially_false() -> None:
    handler = ShutdownHandler()
    assert handler.is_set is False


@pytest.mark.asyncio
async def test_signal_sets_event() -> None:
    handler = ShutdownHandler()
    handler.install()

    assert handler.is_set is False

    # Simulate SIGINT via the loop's signal handler mechanism
    loop = asyncio.get_running_loop()
    loop.call_soon(handler._handle, signal.SIGINT)

    await asyncio.sleep(0.05)
    assert handler.is_set is True


@pytest.mark.asyncio
async def test_wait_returns_after_signal() -> None:
    handler = ShutdownHandler()
    handler.install()

    loop = asyncio.get_running_loop()

    # Schedule the signal after a short delay
    loop.call_later(0.05, handler._handle, signal.SIGTERM)

    await asyncio.wait_for(handler.wait(), timeout=2.0)
    assert handler.is_set is True


@pytest.mark.asyncio
async def test_second_signal_force_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second signal should attempt os._exit(1)."""
    handler = ShutdownHandler()
    handler.install()

    exit_called_with: list[int] = []
    monkeypatch.setattr("agentspype.runner.shutdown.os._exit", exit_called_with.append)

    # First signal — graceful
    handler._handle(signal.SIGINT)
    assert handler.is_set is True
    assert exit_called_with == []

    # Second signal — force exit
    handler._handle(signal.SIGINT)
    assert exit_called_with == [1]
