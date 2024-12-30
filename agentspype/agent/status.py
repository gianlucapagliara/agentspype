from pydantic import BaseModel, ConfigDict


class AgentStatus(BaseModel):
    """Base status class for agents."""

    model_config = ConfigDict()
