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
    prompt += f'{{\n\t"prompt": "{user}",\n\t"name": "'
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

# def generate_structured_call(prompt, schema_definitions, model):

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

end_token = model.encode("<|im_end|>")
parameters = model.encode('", "arguments": {')[0].tolist()
start = time.time()


prompt = genearet_prompt(msg["system"], msg["user"], msg["tools"])
tokens = model.encode(prompt)[0].tolist()

state = "FUNCTION_NAME"
current_generated_string = ""
chosen_function = None
schema_parameters = {}
target_functions = list(list_of_functions.keys())
clean_vocab = build_clean_vocab(model)
print("--- Start Generating Function Name ---")
print(f'{{"prompt": "{msg["user"]}", "name": "', end="", flush=True)
while True:
    logits = model.get_logits_from_input_ids(tokens)
    
    if state == "FUNCTION_NAME":
        allowed_ids = get_allowed_ids_for_prefix(target_functions, current_generated_string, clean_vocab)

    elif state == "PARAM_KEY":
        remaining_keys = [f'"{k}": ' for k in schema_parameters.keys()]
        allowed_ids = get_allowed_ids_for_prefix(remaining_keys, current_generated_string, clean_vocab)

    elif state == "PARAM_VALUE":
        current_type = schema_parameters[current_key]["type"]
        if current_type == "number":
            allowed_ids = get_allowed_ids_for_numbers(clean_vocab)

    masked_logits = apply_logits_mask(logits, allowed_ids)
    next_token = int(np.argmax(masked_logits))
    tokens.append(next_token)
    decoded_token = clean_vocab[next_token]
    current_generated_string += decoded_token
    print(decoded_token, end="", flush=True)
    if state == "FUNCTION_NAME":    
        if current_generated_string in target_functions:
                tokens.extend(model.encode('", "arguments": {')[0].tolist())
                print('", "arguments": {', end="", flush=True)
                schema_parameters = list_of_functions[current_generated_string]["parameters"]
                if not schema_parameters:
                    tokens.extend(model.encode('}')[0].tolist())
                    state = "END"
                else:
                    state = "PARAM_KEY"
                    current_generated_string = ""
    elif state == "PARAM_KEY":
        if current_generated_string in [f'"{k}": ' for k in schema_parameters.keys()]:
            current_key = current_generated_string.split('"')[1] 
            current_generated_string = ""
            state = "PARAM_VALUE"

    elif state == "PARAM_VALUE":
        if "," in decoded_token:
            del schema_parameters[current_key]
            current_generated_string = ""
            state = "PARAM_KEY"
            
        elif "}" in decoded_token:
            print("}", end="", flush=True)
            state = "END" 
    elif state == "END":
        break

   
print("\ntotal:", (time.time() - start) / 60, " minutes")