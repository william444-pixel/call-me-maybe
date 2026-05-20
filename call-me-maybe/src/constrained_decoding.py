from llm_sdk.llm_sdk import Small_LLM_Model
import json
from typing import Any, Dict, List, Optional, Set


def extract_clean_json(text: str) -> Optional[str]:
    """Extracts the first complete balanced JSON object
    enclosed in curly braces.

    Args:
        text (str): The raw text string containing the potential JSON object.

    Returns:
        Optional[str]: The extracted clean JSON string
        if found, otherwise None.
    """
    start = text.find("{")
    if start == -1:
        return None
    count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            count += 1
        if text[i] == "}":
            count -= 1
        if count == 0:
            return text[start:i+1]
    return None


def get_valid_tokens(logits: List[float], valid_id: Set[int]) -> int:
    """Finds the token ID from valid_id that has the highest score in logits.

    Args:
        logits (List[float]): The raw prediction scores from the model.
        valid_id (Set[int]): The set of allowed token IDs.

    Returns:
        int: The selected token ID with the maximum logit score.
    """
    return max(valid_id, key=lambda i: logits[i] if i < len(logits)
               else float('-inf'))


def build_json_valid_ids(vocab: Dict[str, int]) -> Set[int]:
    """Filters the vocabulary to keep only JSON-safe tokens.

    Args:
        vocab (Dict[str, int]): The vocabulary mapping token strings to IDs.

    Returns:
        Set[int]: A set of valid token IDs that contain only safe characters.
    """
    up = "abcdefghijklmnopqrstuvwxyz".upper()
    json_safe = set(
        up + 'abcdefghijklmnopqrstuvwxyz''0123456789*_,.:-+\\/?()[]{}"ĠĊ')
    valid = set()
    for token_str, token_id in vocab.items():
        if token_str and all(c in json_safe for c in token_str):
            valid.add(token_id)
    return valid


def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
    """Loads the raw vocabulary dictionary from the model's tokenizer file.

    Args:
        model (Small_LLM_Model): The model instance providing the
        tokenizer path.

    Returns:
        Dict[str, int]: The raw vocabulary dictionary.
    """
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    raw_vocab = tok_data.get("model", {}).get("vocab", {})
    return dict(raw_vocab)


def build_system_prompt(func: List[Any]) -> str:
    """Builds the system prompt containing
    instructions and available functions.

    Args:
        func (List[Any]): A list of available function definitions.

    Returns:
        str: The formatted system prompt.
    """
    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching \
            functions from the list bellow.",
        "IF NO functions matches the user's intent \
            (even if type match), set name: \"None\".",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",
    ]
    for fn in func:
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f"  -{fn.name}({params}): {fn.description}")
    lines.append(
        '\nOutput ONLY valid JSON: {"name": "<fn>", "parameters":{<arg>}}')
    return "\n".join(lines)
