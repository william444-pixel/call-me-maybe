from llm_sdk.llm_sdk import Small_LLM_Model
import json


def load_vocab(model: Small_LLM_Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, "r") as f:
        tok_data = json.load(f)
    raw_vocab = tok_data.get("model", {}).get("vocab", {})
    return raw_vocab

def build_system_prompt(func):
    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching \
            functions from the list bellow.",
        "IF NO functions matches the user's intent \
            (even if type match).set name: \"None\".",
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
