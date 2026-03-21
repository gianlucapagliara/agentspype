"""Clock support for agentspype — optional integration with chronopype.

All imports from this package require chronopype to be installed.
If chronopype is not available, importing will raise an ImportError
with a helpful installation message.
"""

try:
    from chronopype import BaseClock, ClockConfig, ClockMode
except ImportError as _err:
    raise ImportError(
        "chronopype is required for clock support. "
        "Install it with: pip install agentspype[clock]"
    ) from _err

from agentspype.clock.agent import ClockAgent
from agentspype.clock.configuration import ClockAgentConfiguration
from agentspype.clock.definition import ClockAgentDefinition
from agentspype.clock.listening import ClockListening
from agentspype.clock.publishing import ClockAgentPublishing
from agentspype.clock.state_machine import (
    BasicClockStateMachine,
    ClockStateMachine,
)

__all__ = [
    # Clock classes (re-exported from chronopype)
    "BaseClock",
    "ClockConfig",
    "ClockMode",
    # Agent classes
    "ClockAgent",
    "ClockAgentConfiguration",
    "ClockAgentDefinition",
    "ClockListening",
    "ClockAgentPublishing",
    "ClockStateMachine",
    "BasicClockStateMachine",
]
