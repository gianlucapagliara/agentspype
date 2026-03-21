"""BaseRuntime — generic typed resource registry for agent runners."""

from __future__ import annotations

import logging
import weakref
from typing import Any, TypeVar, overload

logger = logging.getLogger(__name__)

T = TypeVar("T")

_current_runtime: BaseRuntime | None = None


def get_runtime() -> BaseRuntime:
    """Return the current BaseRuntime singleton.

    Raises RuntimeError if the runtime has not been set.
    """
    if _current_runtime is None:
        raise RuntimeError("Runtime not initialized. Call set_runtime() first.")
    return _current_runtime


def set_runtime(runtime: BaseRuntime | None) -> None:
    """Set or clear the current BaseRuntime singleton."""
    global _current_runtime
    _current_runtime = runtime


class BaseRuntime:
    """Abstract resource registry providing typed access to shared resources.

    Resources are stored by string key. Weak references are used for objects
    that support them; otherwise a strong reference is kept. On teardown all
    resources are cleared.
    """

    def __init__(self) -> None:
        self._resources: dict[str, Any] = {}
        self._weak_resources: dict[str, weakref.ref[Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, key: str, resource: Any) -> None:
        """Register a resource under *key*.

        If the resource supports weak references it is stored as a weakref;
        otherwise a strong reference is kept.

        Raises ValueError if a resource is already registered under *key*.
        """
        if key in self._resources or key in self._weak_resources:
            raise ValueError(f"Resource already registered under key '{key}'")

        try:
            self._weak_resources[key] = weakref.ref(
                resource, lambda _ref, _key=key: self._on_weakref_dead(_key)
            )
        except TypeError:
            # Object does not support weakrefs — keep a strong reference.
            self._resources[key] = resource

    @overload
    def get(self, key: str) -> Any: ...

    @overload
    def get(self, key: str, expected_type: type[T]) -> T: ...

    def get(self, key: str, expected_type: type[T] | None = None) -> Any:
        """Retrieve a resource by *key*.

        When *expected_type* is given the returned value is validated against
        that type and a ``TypeError`` is raised on mismatch.

        Raises ``KeyError`` when the key is unknown or the weakref has expired.
        """
        value: Any = None
        if key in self._resources:
            value = self._resources[key]
        elif key in self._weak_resources:
            value = self._weak_resources[key]()
            if value is None:
                del self._weak_resources[key]
                raise KeyError(
                    f"Resource '{key}' is no longer available (garbage collected)"
                )
        else:
            raise KeyError(f"No resource registered under key '{key}'")

        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"Resource '{key}' is {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
        return value

    def has(self, key: str) -> bool:
        """Return ``True`` if a live resource is registered under *key*."""
        if key in self._resources:
            return True
        if key in self._weak_resources:
            if self._weak_resources[key]() is not None:
                return True
            del self._weak_resources[key]
        return False

    async def teardown(self) -> None:
        """Clear all registered resources."""
        self._resources.clear()
        self._weak_resources.clear()
        logger.debug("Runtime resources cleared")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_weakref_dead(self, key: str) -> None:
        """Callback invoked when a weak-referenced resource is collected."""
        self._weak_resources.pop(key, None)
        logger.debug("Weak resource '%s' was garbage collected", key)
