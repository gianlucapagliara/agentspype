from pydantic import BaseModel, ConfigDict


class AgentConfiguration(BaseModel):
    """Base configuration class for agents."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=False)
