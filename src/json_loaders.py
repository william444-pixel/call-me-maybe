import json
from typing import List
from src.rules.function_defi import FunctionDefinition
from src.rules.prompts import Prompts


def load_function_definition(path: str) -> List[FunctionDefinition]:
    """Loads and validates a collection of system tool
    configurations from a JSON file.

    Args:
        path (str): The explicit file path pointing to the
        function definitions JSON asset.

    Returns:
        List[FunctionDefinition]: A list of validated Pydantic tool structures.

    Raises:
        Exception: If the file asset cannot be discovered
        or contains malformed JSON syntax.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return [FunctionDefinition(**item) for item in data]
    except FileNotFoundError:
        raise Exception(f"file not found {path}")
    except json.JSONDecodeError:
        raise Exception(f"json format invalid {path}")


def load_prompt(path: str) -> List[Prompts]:
    """Loads and parses input user evaluation queries
    from a structured dataset file.

    Args:
        path (str): The explicit file path pointing
        to the user prompts JSON asset.

    Returns:
        List[Prompts]: A list of validated Pydantic query instances.

    Raises:
        Exception: If the file asset cannot be discovered or contains
        malformed JSON syntax.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return [Prompts(**item) for item in data]
    except FileNotFoundError:
        raise Exception(f"file not found {path}")
    except json.JSONDecodeError:
        raise Exception(f"json format invalid {path}")
