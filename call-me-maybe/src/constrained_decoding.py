from llm_sdk.llm_sdk import Small_LLM_Model
import json
def extract_clean_json(text: str):
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
def get_valid_tokens(logits, valid_id):
    return max(valid_id, key=lambda i:logits[i] if i < len(logits) else float('-inf'))

def build_json_valid_ids(vocab):
    json_safe = set('abcdefghijklmnopqrstuvwxyz''0123456789*_,:-+/\?()[]{}"ĠĊ')
    valid = set()
    for token_str, token_id in vocab.items():
        if token_str and all(c in json_safe for c in token_str):
            valid.add(token_id)
    return valid

def load_vocab(model: Small_LLM_Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    raw_vocab = tok_data.get("model", {}).get("vocab", {})
    return raw_vocab


def build_system_prompt(func):
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
    lines.append('\nOutput ONLY valid JSON: {"name": "<fn>", "args":{<arg>}}')
    return "\n".join(lines)
