from pydantic import BaseModel


class Prompts(BaseModel):
    """
    Represents a single raw input user query object loaded from dataset
    configurations.
    """
    prompt: str
