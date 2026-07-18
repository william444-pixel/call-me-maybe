from llm_sdk.llm_sdk import Small_LLM_Model
import json
# from typing import Any, Dict, List, Optional, Set
import numpy as np


def get_mask_logits(allowed_ids: list[int], logits: list[float])\
      -> list[float]:
    mask_logits = np.full_like(logits, -np.inf)
    for allowed_id in allowed_ids:
        mask_logits[allowed_id] = logits[allowed_id]
    return mask_logits  # type: ignore[return-value]


def get_allowed_ids_for_numbers(
    clean_vocab: dict[int, str], is_last: bool
) -> list[int]:
    """Return token IDs that are safe to emit while generating a numeric
    JSON value.  When is_last=True the value is terminated by `}`;
    otherwise it is terminated by `,`."""
    allowed_ids: list[int] = []
    allowed_chars = set("0123456789.-}") if is_last else set("0123456789.-,")
    for token_id, token_text in clean_vocab.items():
        if (not token_text or token_text.count("}") > 1
                or token_text.count(",") > 1):
            continue
        if all(char in allowed_chars for char in token_text):
            allowed_ids.append(token_id)
    return allowed_ids


def get_allowed_ids_for_strings(
    clean_vocab: dict[int, str], is_last: bool
) -> list[int]:
    """Return token IDs that are safe to emit while generating a string
    JSON value.  After the closing quote only structural characters are
    allowed.  When is_last=True the value ends with `}`; otherwise `,`."""
    allowed_ids: list[int] = []
    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        if "\n" in token_text or "\r" in token_text:
            continue
        unescaped_text = token_text.replace('\\"', "")
        if '"' in unescaped_text:
            after_quote = unescaped_text[unescaped_text.find('"') + 1:]
            allowed_closing = "} " if is_last else ", "
            if any(char not in allowed_closing for char in after_quote):
                continue
            if after_quote.count("}") > 1 or after_quote.count(",") > 1:
                continue
        allowed_ids.append(token_id)
    return allowed_ids


def build_clean_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocabulary = dict()
    with open(model.get_path_to_vocab_file(), "r") as f:
        vocabulary = json.load(f)
    clean_vocab: dict[int, str] = {}
    for _, token_id in vocabulary.items():
        clean_vocab[token_id] = model.decode(token_id)
    return clean_vocab


def get_tokens_allowed_ids(
    clean_vocab: dict[int, str], gen: str, list_target: list[str]
) -> list[int]:
    allowed_ids = []
    for token_id, token_text in clean_vocab.items():
        text_target = gen + token_text
        for target in list_target:
            if target.startswith(text_target):
                allowed_ids.append(token_id)
                break
    return allowed_ids
