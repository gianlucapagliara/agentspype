"""Stateless utilities for launch identification.

:func:`get_launch_hash` produces a deterministic identifier for a specific
launch, derived from the configuration hash and start time.
"""

from __future__ import annotations

import hashlib


def get_launch_hash(config_hash: str, start_time: float) -> str:
    """Return a deterministic SHA-256 hash identifying a specific launch.

    Parameters
    ----------
    config_hash:
        The hash of the configuration used for this launch.
    start_time:
        The Unix timestamp when the launch started.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(f"{config_hash}_{start_time}".encode()).hexdigest()
