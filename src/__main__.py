import json
import argparse
import os
import time
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
from src.json_loaders import load_function_definition, load_prompt
from src.constrained_decoding import (
    get_mask_logits,
    get_allowed_ids_for_numbers,
    get_allowed_ids_for_strings,
    build_clean_vocab,
    get_tokens_allowed_ids,
)

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

def main():
    args = arg_parser()
    total_start = time.perf_counter()
    final_results = []
    list_alloweds_of_functions = []
    schema_parameters = {}

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
                keys = [f'"{key}":' for key in schema_parameters.keys()]
                allowed_ids = get_tokens_allowed_ids(clean_vocab, gen, keys)
            elif state == "PARAM_VALUE":
                is_last_param = len(schema_parameters) == 1
                if current_key in schema_parameters:
                    param_type = schema_parameters[current_key].type
                    if param_type == "string":
                        allowed_ids = get_allowed_ids_for_strings(
                            clean_vocab, is_last_param
                        )
                    elif param_type in ("number", "integer", "int"):
                        allowed_ids = get_allowed_ids_for_numbers(
                            clean_vocab, is_last_param
                        )
                    

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
                    schema_parameters = list_of_functions[matched_function_name].parameters.copy()
                    gen = ""
                    state = "PARAM_KEY"

            elif state == "PARAM_KEY":
                if gen in [f'"{key}":' for key in schema_parameters.keys()]:
                    current_key = gen.split('"')[1]
                    gen = ""
                    state = "PARAM_VALUE"

            elif state == "PARAM_VALUE":
                current_type = schema_parameters[current_key].type
                is_value_complete = False
                if current_type in ["number", "boolean", "integer"]:
                    if "," in gen or "}" in gen:
                        is_value_complete = True
                    ending_part = gen
                elif current_type == "string":
                    clean_gen = gen.replace('\\"', "")
                    ending_part = clean_gen.split('"')[-1]
                    if clean_gen.count('"') >= 2 and (
                        "," in clean_gen.split('"')[-1]
                        or "}" in clean_gen.split('"')[-1]
                    ):
                        is_value_complete = True
                if is_value_complete:
                    if "," in gen:
                        del schema_parameters[current_key]
                        gen = ""
                        state = "PARAM_KEY" if schema_parameters else "END"
                    elif "}" in gen:
                        if ending_part.count("}") == 1:
                            tokens.extend(end_obj_injection)
                        state = "END"

        # --- POST-PROCESSING & EXTRACTION LAYER ---
        result_raw = model.decode(tokens)

        try:
            json_start_index = result_raw.find(f'"name":"{matched_function_name}"')
            if json_start_index == -1:
                raise ValueError("JSON start object not found.")

            clean_json_str = f'{{"prompt":{raw_prompt_text},'
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
                {"prompt": json.loads(raw_prompt_text), "name": None, "parameters": {}}
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