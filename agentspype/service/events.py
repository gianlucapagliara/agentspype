"""Re-export QueueEventSubscriber from eventspype for backward compatibility.

The agentspype service layer uses ``QueueEventSubscriber`` (from eventspype
>=1.2.0) as its event bridge between synchronous eventspype publications and
async SSE consumer queues.
"""

from __future__ import annotations

from eventspype.sub.queue import QueueEventSubscriber

__all__ = ["QueueEventSubscriber"]
