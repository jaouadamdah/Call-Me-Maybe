import time
from llm_sdk import Small_LLM_Model
import numpy as np
import json
from pprint import pprint
import copy

model = Small_LLM_Model()


def genearet_prompt(system: str, user: str, tools: str):
    prompt = f"<|im_start|>system\n{system}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:"
    prompt += f"\n<tools>\n{tools}\n</tools>"
    prompt += "\n\nFor each function call, return a json object with user prompt, function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"prompt\": <user-prompt>, \"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
    prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt += f'{{\n\t"prompt": "{user}",\n\t"name": "'
    return prompt


tokens_of_functions = {}
with open("/goinfre/jamdah/call_me_maybe/data/input/functions_definition.json") as fun, open("/goinfre/jamdah/call_me_maybe/data/input/function_calling_tests.json") as p:
    objs = json.load(fun)
    for obj in objs:
        tokens_of_functions[obj["name"]] = model.encode(obj["name"])[
            0].tolist()
    prompts = json.load(p)
    functions = "\n".join(json.dumps(line) for line in objs)

msg = {
    'system': 'You are an AI assistant that responds to tools queries. You\
        should reply with the needed function or tool.',
    'user': "What is the sum of 2 and 3?",
    'tools': functions
}

end_token = model.encode('<|im_end|>')
parameters = model.encode('",\n\t"arguments": {\n\t\t"')[0].tolist()
start = time.time()
# arr = [1,2,3]
# for token in model.encode('fn_substitute_string_wiith_regex')[0].tolist():
#     tokens_of_functions_used = {}

#     for name, values in tokens_of_functions.items():
#         if token == values[0]:
#             values.pop(0)
#             tokens_of_functions_used[name] = values
#     pprint(tokens_of_functions_used)
#     if len(tokens_of_functions_used) == 1:
#         arr.extend(*tokens_of_functions_used.values())
#         break
# pprint(arr)

# exit()
for p in prompts:
    prompt = genearet_prompt(msg['system'], p, msg['tools'])
    tokens = model.encode(prompt)[0].tolist()
    max_tokens = 1200
    next_token = None

    print(f'{{\n\t"prompt": "{p["prompt"]}",\n\t"name": "', end='', flush=True)
    functions_to_use = copy.deepcopy(tokens_of_functions)

    while max_tokens and next_token != end_token:
        next_token = np.array(model.get_logits_from_input_ids(tokens)).argmax()
        tokens_of_functions_used = {}
        for name, values in functions_to_use.items():
            if next_token == values[0]:
                values.pop(0)
                tokens_of_functions_used[name] = values
        if len(tokens_of_functions_used) == 1:
            for values in tokens_of_functions_used.values():
                tokens.extend([next_token, *values, *parameters])
                print(model.decode([next_token, *values, *parameters]),
                    end='', flush=True)
        else:
            tokens.append(next_token)
            print(model.decode(next_token), end='', flush=True)

        max_tokens -= 1

print('\ntotal:', (time.time() - start) // 60)
