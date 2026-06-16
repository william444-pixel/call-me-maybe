import argparse
import json
import numpy as np
import os
from typing import Any, Dict, List
from src.json_loaders import load_function_definition, load_prompt
from src.constrained_decoding import (
    build_json_valid_ids,
    build_system_prompt,
    extract_clean_json,
    get_valid_tokens,
    load_vocab,
    get_numeric_mask_numpy,
)
from llm_sdk.llm_sdk import Small_LLM_Model


def arg_parser() -> argparse.Namespace:
    """Parses command line arguments for the function calling
    generation script.

    Returns:
        argparse.Namespace: The parsed command line arguments containing paths
            for input, functions definition, output, and the model name.
    """
    parse = argparse.ArgumentParser(
        description="transalte from prompt to function calls..."
    )
    parse.add_argument(
        "--input", type=str, default="data/input/function_calling_tests.json"
    )
    parse.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parse.add_argument(
        "--output", type=str, default="data/output/function_results_tests.json"
    )
    parse.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    return parse.parse_args()


def main() -> None:
    """Main execution function that runs the constrained decoding pipeline

    to translate user prompts into valid, structured function calling
    JSON outputs.
    """
    print("Starting functions and prompts")
    args = arg_parser()
    func = load_function_definition(args.functions_definition)
    functions = func[0]
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
    all_results: List[Dict[str, Any]] = []
    for p in prompts:
        line_clean_json = ""
        prompt = p.prompt
        schema_parameters = {}
        print(f"Processing prompt: {prompt}")
        full_prompt = f'{system}\n\nUser prompt: {prompt}\nAssistant: {{"name": '
        tokens = model.encode(full_prompt)[0].tolist()
        clean_json = None
        while True:
            is_finish = False
            logits = np.array(model.get_logits_from_input_ids(tokens))
            text = model.decode(tokens)
            current_param_is_numeric = False
            schema_parameters = functions.parameters.copy()
            if is_finish:
                for param_name, param_type in schema_parameters.items():
                    print(param_type.type.lower())
                    print(param_name)
                    if f'"{param_name}": ' in text and param_type.type.lower() in [
                        "number",
                        "float",
                        "integer",
                    ]:
                        after_param = text.split(f'"{param_name}":')[-1]
                        del schema_parameters[param_name]
                        current_param_is_numeric = True
            if current_param_is_numeric:
                print("is a number")
                masked_logits = get_numeric_mask_numpy(vocab)
                next_id = get_valid_tokens(logits, masked_logits)
            else:
                print("the next parameter")
                next_id = get_valid_tokens(logits, valid_ids)
            tokens.append(next_id)
            text = model.decode(tokens)
            if "}" in text[-1]:
                print(text)
                break
            if "," in text:
                is_finish = True

            # if current_param_is_numeric and param_type.type.lower() in [
            #     "number",
            #     "float",
            # ]:
            # Get the updated text after appending the new token
            # updated_after_param = text.split(f'"{param_name}":')[-1].strip()

            # If the new token is a comma or a closing brace, it means generation of this number IS DONE
            # if "," in updated_after_param or "}" in updated_after_param:

            # Check if the number was generated WITHOUT a decimal point (e.g., "2" instead of "2.5")
            # We look at the text BEFORE the comma/brace was added
            # if "." not in after_param:
            #     # Pop the structural token (comma or brace) temporarily
            #     last_structural_token = all_gen.pop()

            #     # Force insert the ".0" safely at the end of the number
            #     all_gen.extend(model.encode(".0")[0].tolist())

            #     # Put back the structural token to close the parameter correctly
            #     all_gen.append(last_structural_token)

            # Update the final text string
            # text = model.decode(all_gen)
            print(text[text.find(f"prompt: {prompt}"):], end="\n", flush=True)
            # clean_json = extract_clean_json(text)
            # if clean_json:
            #     try:
            #         line_clean_json = json.loads(clean_json)
            #         print(line_clean_json)
            #         break
            #     except Exception:
            #         pass
        print("clean: ", clean_json)
        if not clean_json or not isinstance(line_clean_json, dict):
            print("no clean")
            tokens.extend(model.encode("}}")[0].tolist())
            logits = model.get_logits_from_input_ids(tokens)
            text = model.decode(tokens)
            print(text)
            clean_json = extract_clean_json(text)
            print(clean_json)
            line_clean_json = json.loads(clean_json)
        all_results.append(
            {
                "prompt": prompt,
                "name": line_clean_json.get("name", "none"),
                "parameters": line_clean_json.get("parameters", {}),
            }
        )
        if line_clean_json.get("name") != "none":
            print("[succes]")
        else:
            print(" -> [ERROR] Could not generate function call")

    clean_output = [result for result in all_results if result["name"] != "none"]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(clean_output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("User stop the program")
    except Exception as e:
        print(f"Error: {e}")
