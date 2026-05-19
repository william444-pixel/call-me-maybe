import argparse
from src.json_loaders import load_function_definition, load_prompt
from src.constrained_decoding import build_system_prompt, load_vocab, build_json_valid_ids, get_valid_tokens, extract_clean_json
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json
import os

def arg_parser():
    parse = argparse.ArgumentParser(description="transalte from prompt to function calls...")
    parse.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )
    parse.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )
    parse.add_argument(
        "--output",
        type=str,
        default="data/output/function_results_tests.json"
    )
    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B"
    )
    return parse.parse_args()


def main():
    print("Starting functions and prompts")
    args = arg_parser()
    func = load_function_definition(args.functions_definition)
    if not func:
        raise Exception("No Functions definition found")
    prompts = load_prompt(args.input)
    if not prompts:
        raise Exception("No Prompt found")
    print("Building system prompts")
    system = build_system_prompt(func)
    print(f"loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except Exception:
        raise Exception(f"Model {args.model} not found")
    vocab = load_vocab(model)
    valid_ids = build_json_valid_ids(vocab)
    all_results = []
    for p in prompts:
        prompt = p.prompt
        print(f"Processing prompt: {prompt}")
        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        input_ids = model.encode(full_prompt)
        gen_ids = input_ids[0].tolist()

        clean_json = None
        all_gen = []
        all_gen.extend(model.encode('{"name": "')[0].tolist())
        for _ in range(55):
            logits = model.get_logits_from_input_ids(gen_ids + all_gen)
            next_id = get_valid_tokens(logits, valid_ids)
            all_gen.append(next_id)
            text = model.decode(all_gen)
            print(text, end="\n", flush=True)
            clean_json = extract_clean_json(text)
            if clean_json:
                try:
                  line_clean_json = json.loads(clean_json)
                  break
                except Exception:
                 pass
        if not clean_json:
            line_clean_json = {"name": "none", "parameters": {}}
        all_results.append({
            "prompt": prompt,
            "name": line_clean_json.get("name", "none"),
            "parameters": line_clean_json.get("parameters", ())
        })
        if line_clean_json.get("name") != "none":
            print(f"[succes]")
        else:
            print(f" -> [ERROR] Could not generate function call")


    clean_output = [result for result in all_results if result["name"] != "none"]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean_output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"User stop the program")
    except Exception as e:
        print(f"Error: {e}")
