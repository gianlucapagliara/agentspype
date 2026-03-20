import logging
import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from eventspype.sub.multisubscriber import MultiSubscriber

if TYPE_CHECKING:
    from agentspype.agent.agent import Agent


class AgentListening(MultiSubscriber):
    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self._agent = weakref.ref(agent)

    # === Properties ===

    @property
    def agent(self) -> "Agent":
        agent = self._agent()
        if agent is None:
            raise RuntimeError("Agent has been deactivated")
        return agent

    def logger(self) -> logging.Logger:
        return self.agent.logger()

    # === Event Definition Support ===

    @classmethod
    def get_event_definitions(cls) -> dict[str, Any]:
        """Get all event subscriptions defined in this class."""
        # This is a placeholder implementation - the actual implementation
        # would depend on how eventspype stores subscription information
        # For visualization purposes, we'll look for methods that might be callbacks
        subscriptions = {}
        for name, value in cls.__dict__.items():
            if (
                callable(value)
                and not name.startswith("_")
                and name not in ["subscribe", "unsubscribe", "logger"]
            ):
                subscriptions[name] = {
                    "callback": value,
                    "callback_name": name,
                    # These would be determined by actual eventspype implementation
                    "event_tag": getattr(value, "event_tag", None),
                    "publisher_class": getattr(value, "publisher_class", None),
                }
        return subscriptions

    # === Subscriptions ===

    @abstractmethod
    def subscribe(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unsubscribe(self) -> None:
        raise NotImplementedError
