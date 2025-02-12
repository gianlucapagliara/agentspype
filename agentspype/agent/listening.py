import logging
import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING

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

    # === Subscriptions ===

    @abstractmethod
    def subscribe(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unsubscribe(self) -> None:
        raise NotImplementedError
