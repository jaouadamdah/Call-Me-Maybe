import time
from llm_sdk import Small_LLM_Model
import numpy as np
import json
from pprint import pprint

model = Small_LLM_Model()


def genearet_prompt(system: str, user: str, tools: str):
    prompt = f"<|im_start|>system\n{system}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:"
    prompt += f"\n<tools>\n{tools}\n</tools>"
    prompt += '\n\nFor each function call, return a json object with user prompt, function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"prompt": <user-prompt>, "name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
    prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt += f'{{"prompt": "{user}", "name": "'
    return prompt

def build_clean_vocab(model):
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocabulary = json.load(f)
    clean_vocab = {}
    for _, token_id in vocabulary.items():
        clean_text = model.decode([token_id])
        clean_vocab[token_id] = clean_text
    return clean_vocab

def apply_logits_mask(logits, allowed_ids):
    masked_logits = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        masked_logits[token_id] = logits[token_id]
    return masked_logits

def get_allowed_ids_for_prefix(target_list: list[str], current_string: str, clean_vocab: dict):
    allowed_ids: list = []

    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        text_target: str = current_string + token_text
        for target in target_list:
            if target.startswith(text_target) or text_target.startswith(target):
                allowed_ids.append(token_id)
                break
    return allowed_ids

def get_allowed_ids_for_numbers(clean_vocab: dict[int, str]):
    allowed_ids: list = []

    allowed_chars = set("0123456789.-,} ")
    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        if all(char in allowed_chars for char in token_text):
            allowed_ids.append(token_id)
    return allowed_ids

tokens_of_functions = {}
list_of_functions = {}

with (
    open("/goinfre/alamliti/Call-Me-Maybe/data/input/functions_definition.json") as fun,
    open("/goinfre/alamliti/Call-Me-Maybe/data/input/function_calling_tests.json") as p,
):
    objs = json.load(fun)
    for obj in objs:
        tokens_of_functions[obj["name"]] = model.encode(obj["name"])[0].tolist()
        list_of_functions[obj["name"]] = obj
    prompts = json.load(p)
    functions = "\n".join(json.dumps(line) for line in objs)

msg = {
    "system": "You are an AI assistant that responds to tools queries. You\
        should reply with the needed function or tool.",
    "user": "What is the sum of 2 and 3?",
    "tools": functions,
}

start = time.time()
# FUNCTION_NAME | PARAM_KEY | PARAM_VALUE
def generate_structured_call(prompt, schema_definitions, model):
    prompt = genearet_prompt(msg["system"], prompt, schema_definitions)
    tokens = model.encode(prompt)[0].tolist()

    state = "FUNCTION_NAME"
    current_generated_string = ""
    schema_parameters = {}
    target_functions = list(list_of_functions.keys())
    clean_vocab = build_clean_vocab(model)

    while True:
        logits = model.get_logits_from_input_ids(tokens)
        
        if state == "FUNCTION_NAME":
            allowed_ids = get_allowed_ids_for_prefix(target_functions, current_generated_string, clean_vocab)


    return tokens
tokens = generate_structured_call("What is the sum of 222, 33.3?", functions, model)
print(model.decode(tokens), end="", flush=True)  
print("\ntotal:", (time.time() - start) / 60, " minutes")