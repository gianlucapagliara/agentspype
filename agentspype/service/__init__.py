"""Service layer for agentspype — requires the ``[service]`` extra (processpype).

Usage::

    from agentspype.service import AgentRunnerService, AgentRunnerServiceConfiguration

The models submodule (``AgentSummaryModel``, etc.) is always available.
The processpype-dependent classes (``AgentRunnerService``, ``AgentRunnerRouter``,
etc.) are only importable when processpype is installed.
"""

from agentspype.service.models import (
    AgentDetailModel,
    AgentSummaryModel,
    HealthModel,
    RunnerStatusModel,
)

__all__ = [
    "AgentDetailModel",
    "AgentSummaryModel",
    "HealthModel",
    "RunnerStatusModel",
]

try:
    from agentspype.service.configuration import AgentRunnerServiceConfiguration
    from agentspype.service.events import QueueEventSubscriber
    from agentspype.service.manager import AgentRunnerManager
    from agentspype.service.router import AgentRunnerRouter
    from agentspype.service.service import AgentRunnerService

    __all__ += [
        "AgentRunnerService",
        "AgentRunnerServiceConfiguration",
        "AgentRunnerManager",
        "AgentRunnerRouter",
        "QueueEventSubscriber",
    ]
except ImportError:
    pass
