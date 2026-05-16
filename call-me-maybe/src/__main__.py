import argparse
from src.json_loaders import load_function_definition, load_prompt
from src.constrained_decoding import build_system_prompt, load_vocab
from llm_sdk.llm_sdk import Small_LLM_Model


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
    prompt = load_prompt(args.input)
    if not prompt:
        raise Exception("No Prompt found")
    print("Building system prompts")
    system = build_system_prompt(func)
    print(system)
    print(f"loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        raise Exception(f"Model {args.model} not found")
    vocab = load_vocab(model)
    print(vocab)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
