from typing import Dict
from pydantic import BaseModel


class Param(BaseModel):
    """
    Attributes:
        type (str): The primitive data type constraint of the argument
        (e.g., 'string', 'int').
    """
    type: str


class ReturnType(BaseModel):
    """
    Attributes:
        type (str): The execution datatype yielded post-evaluation.
    """
    type: str


class FunctionDefinition(BaseModel):
    """Defines the rigid operational structure of an executable system tool.

    This layout enforces precise structural generation constraints onto the LLM
    during semantic tool matching processes.
    """
    name: str
    description: str
    parameters: Dict[str, Param]
    returns: ReturnType
