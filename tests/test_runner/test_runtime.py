"""Tests for BaseRuntime resource registry."""

from __future__ import annotations

import gc

import pytest

from agentspype.runner.runtime import BaseRuntime, get_runtime, set_runtime


class _DummyResource:
    """A simple resource that supports weakrefs."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


def test_register_and_get() -> None:
    rt = BaseRuntime()
    res = _DummyResource(42)
    rt.register("my_key", res)

    assert rt.get("my_key") is res


def test_get_with_type_check() -> None:
    rt = BaseRuntime()
    res = _DummyResource(7)
    rt.register("typed", res)

    result = rt.get("typed", _DummyResource)
    assert isinstance(result, _DummyResource)
    assert result.value == 7


def test_get_with_wrong_type_raises() -> None:
    rt = BaseRuntime()
    res = _DummyResource()
    rt.register("typed", res)

    with pytest.raises(TypeError, match="expected str"):
        rt.get("typed", str)


def test_register_duplicate_key_raises() -> None:
    rt = BaseRuntime()
    res = _DummyResource()
    rt.register("dup", res)

    with pytest.raises(ValueError, match="already registered"):
        rt.register("dup", _DummyResource())


def test_get_missing_key_raises() -> None:
    rt = BaseRuntime()

    with pytest.raises(KeyError, match="No resource"):
        rt.get("nonexistent")


# ---------------------------------------------------------------------------
# has
# ---------------------------------------------------------------------------


def test_has_returns_true_for_registered() -> None:
    rt = BaseRuntime()
    res = _DummyResource()
    rt.register("x", res)
    assert rt.has("x") is True


def test_has_returns_false_for_missing() -> None:
    rt = BaseRuntime()
    assert rt.has("x") is False


# ---------------------------------------------------------------------------
# strong reference fallback (int does not support weakref)
# ---------------------------------------------------------------------------


def test_register_non_weakrefable_resource() -> None:
    rt = BaseRuntime()
    rt.register("count", 42)
    assert rt.get("count") == 42


# ---------------------------------------------------------------------------
# weakref cleanup
# ---------------------------------------------------------------------------


def test_weakref_cleanup_after_gc() -> None:
    rt = BaseRuntime()
    res = _DummyResource(99)
    rt.register("weak", res)

    assert rt.has("weak")

    # Drop the only strong reference and collect
    del res
    gc.collect()

    assert rt.has("weak") is False


def test_get_expired_weakref_raises() -> None:
    rt = BaseRuntime()
    res = _DummyResource()
    rt.register("gone", res)
    del res
    gc.collect()

    # The weakref finalizer already removed the key, so get raises KeyError
    with pytest.raises(KeyError):
        rt.get("gone")


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teardown_clears_resources() -> None:
    rt = BaseRuntime()
    res_a = _DummyResource()
    rt.register("a", res_a)
    rt.register("b", 123)
    assert rt.has("a") and rt.has("b")

    await rt.teardown()

    assert rt.has("a") is False
    assert rt.has("b") is False


# ---------------------------------------------------------------------------
# module-level singleton
# ---------------------------------------------------------------------------


def test_get_runtime_before_set_raises() -> None:
    # Ensure clean state
    set_runtime(None)
    with pytest.raises(RuntimeError, match="not initialized"):
        get_runtime()


def test_set_and_get_runtime() -> None:
    rt = BaseRuntime()
    set_runtime(rt)
    try:
        assert get_runtime() is rt
    finally:
        set_runtime(None)
