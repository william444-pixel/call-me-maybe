from pydantic import BaseModel
from typing import Dict

class Param(BaseModel):
    type: str

class ReturnType(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Param]
    returns: ReturnType
