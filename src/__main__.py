import json
import argparse
import os
import time
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
from src.json_loaders import load_function_definition, load_prompt
from pprint import pprint

def arg_parser() -> argparse.Namespace:
    parse = argparse.ArgumentParser(
        description="translate from prompt to function calls..."
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
):
    allowed_ids = []
    for token_id, token_text in clean_vocab.items():
        text_target = gen + token_text
        for target in list_target:
            if target.startswith(text_target):
                allowed_ids.append(token_id)
                break
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


def is_valid_numeric_token(
    token_text: str, gen: str, remaining_params: list
) -> bool:
    full_str = gen + token_text
    if full_str.endswith(","):
        if len(remaining_params) == 0:
            return False
        num_part = full_str[:-1]
    elif full_str.endswith("}"):
        num_part = full_str[:-1]
    else:
        num_part = full_str

    if not num_part:
        return len(gen) > 0
    if not all(c in "0123456789.-" for c in num_part):
        return False
    if num_part.count(".") > 1 or num_part.count("-") > 1:
        return False
    return True


def get_mask_logits(allowed_ids, logits):
    mask_logits = np.full_like(logits, -np.inf)
    for allowed_id in allowed_ids:
        mask_logits[allowed_id] = logits[allowed_id]
    return mask_logits


def is_currently_escaped(s):
    count = 0
    for char in reversed(s):
        if char == '\\':
            count += 1
        else:
            break
    return count % 2 == 1


def main():
    args = arg_parser()
    total_start = time.perf_counter()
    final_results = []
    list_alloweds_of_functions = []
    schema_parametres = {}

    functions_tools_list = load_function_definition(args.functions_definition)
    prompts_list = load_prompt(args.input)
    
    for funcobj in functions_tools_list:
        func_name = funcobj.name
        list_alloweds_of_functions.append(func_name)
        
    list_of_functions = {fn.name: fn for fn in functions_tools_list}
    
    model = Small_LLM_Model()
    clean_vocab = build_clean_vocab(model)

    end_obj_injection = model.encode("}")[0].tolist()
    parametre_injection = model.encode('","parameters":{')[0].tolist()

    for prompt_obj in prompts_list:
        raw_prompt_text = json.dumps(prompt_obj.prompt)
        print(f"\n🚀 Processing prompt: {raw_prompt_text}")

        system_lines = list()
        for fn in functions_tools_list:
            params = ", ".join(name for name, _ in fn.parameters.items())
            system_lines.append(f"{fn.name}({params}):{fn.description}")
        tools = "\n".join(system_lines)
        prompt = "Tools:\n"
        prompt += f"{tools}\n"
        prompt += "- Be extremely precise with strings and regex patterns\n"
        prompt += 'Example:\n{"name":"function-name","parameters":<arguments>}\n'
        prompt += f"User:{raw_prompt_text}\nAssistent:\n"
        prompt += '{"name":"'

        tokens: list = model.encode(prompt)[0].tolist()
        gen = ""
        state = "FUNCTION_NAME"
        
        matched_function_name = ""
        remaining_parameters = []
        current_key = ""

        while state != "END":
            logits = model.get_logits_from_input_ids(tokens)
            allowed_ids = []

            if state == "FUNCTION_NAME":
                allowed_ids = get_tokens_allowed_ids(
                    clean_vocab, gen, list_alloweds_of_functions
                )
            elif state == "PARAM_KEY":
                keys = [f'"{key}":' for key in schema_parametres.keys()]
                allowed_ids = get_tokens_allowed_ids(clean_vocab, gen, keys)
            elif state == "PARAM_VALUE":
                for key, type in schema_parametres.items():

                    allowed_ids = get_tokens_allowed_ids(clean_vocab, gen, ['":'])

            masked_logits = get_mask_logits(allowed_ids, logits)
            next_token = int(np.argmax(masked_logits))
            tokens.append(next_token)
            gen += clean_vocab[next_token]


            print(f"[{state}] -> Added: {repr(clean_vocab[next_token])} | Current gen: '{gen}'")
            # --- DFA STATE TRANSITION MACHINE ---
            if state == "FUNCTION_NAME":
                if gen in list_alloweds_of_functions:
                    tokens.extend(parametre_injection)
                    matched_function_name = gen
                    schema_parametres = list_of_functions[matched_function_name].parameters
                    gen = ""
                    state = "PARAM_KEY"

            elif state == "PARAM_KEY":
                if 

            elif state == "PARAM_VALUE":
                if gen in remaining_parameters:
                    current_key = gen
                    remaining_parameters.remove(gen)
                    gen = ""
                    state = "PARAM_KEY"

        # --- POST-PROCESSING & EXTRACTION LAYER ---
        result_raw = model.decode(tokens)

        try:
            json_start_index = result_raw.find(f'"name":"{matched_function_name}"')
            if json_start_index == -1:
                raise ValueError("JSON start object not found.")

            safe_user_prompt_json = json.dumps(raw_prompt_text)
            clean_json_str = f'{{"prompt":{safe_user_prompt_json},'
            clean_json_str += result_raw[json_start_index:]

            parsed_json = json.loads(clean_json_str)
            func_name = parsed_json.get("name")

            params_dict = parsed_json.get("parameters", {})
            if func_name in list_of_functions:
                func_def = list_of_functions[func_name]
                for param_key, param_value in params_dict.items():
                    param_spec = func_def.parameters.get(param_key)
                    if param_spec:
                        expected_type = param_spec.type
                        try:
                            if expected_type == "number":
                                params_dict[param_key] = float(param_value)
                            elif expected_type in ("integer", "int"):
                                params_dict[param_key] = int(param_value)
                        except (ValueError, TypeError):
                            pass

            final_results.append(parsed_json)
            print(f"✨ Successfully generated: {parsed_json}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[-] CRITICAL ERROR on prompt: {raw_prompt_text}")
            print(f"Error details: {e}")
            final_results.append(
                {"prompt": raw_prompt_text, "name": None, "parameters": {}}
            )

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n[+] Processing Complete!")
    print(
        "[+] Total execution time: ",
        f"{((time.perf_counter() - total_start) / 60):.2f} minutes",
    )
    print(f"[+] Results successfully saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser stopped the program")
    except Exception as e:
        print(f"Error: {e}")