from src.rules.function_defi import FunctionDefinition
from src.rules.prompts import Prompts
import json


def load_function_definition(path: str):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return [FunctionDefinition(**item) for item in data]
    except FileNotFoundError:
        raise Exception(f"file not found {path}")
    except json.JSONDecodeError:
        raise Exception(f"json format invalid {path}")


def load_prompt(path: str):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return [Prompts(**item) for item in data]
    except FileNotFoundError:
        raise Exception(f"file not found {path}")
    except json.JSONDecodeError:
        raise Exception(f"json format invalid {path}")
