from llm_sdk import Small_LLM_Model
from typing import Any
import json
import os

from .tokenizer import Tokenizer
from .parser import read_args
from .models import Prompts, FuncDefs
from .generate import generate_record

# import json
# with open('data/input/functions_definition.json', 'r') as f:
#     raw = json.load(f)
# defs = FuncDefs.model_validate(raw)

system_prompt = ("<|im_start|>system\nYou are a helpful assistant that will perform function calling.\n"
                 "Your job is to first understand what functions you can use, and then to "
                 "select a correct one based on prompt. "
                 "You need to provide a valid JSON, in this format: "
                 '{"function": "function name", "arguments":{"name": "value"}}\n'
                 "Remember to put in arguments correctly. Pay utmost attention to regex.\n"
                 )

# system_prompt = ("You are a helpful assistant that will perform function calling.\n"
#                  "Your job is to first understand what functions you can use, and then to "
#                  "select a correct one based on prompt. "
#                  "You need to provide a valid JSON, in this format: "
#                  '{"function": "function name", "arguments":{"name": "value"}}\n'
#                  "Remember to put in arguments correctly. Pay utmost attention to regex.\n"
#                  )

# def _make_prompt(input_prompt: str, defs: str, ) -> str:
#         system_prompt = ("You are a helpful assistant that will perform function calling.\n"
#                          "Your job is to first understand what functions you can use, and then to "
#                          "select a correct one based on prompt. "
#                          "You need to provide a valid JSON, in this format: "
#                          '\{"function_name"\}: str, name: str, and parameters: dict.\n'
#                          "Remember to put in parameters correctly. Pay utmost attention to regex.\n"
#                          "Allowed functions are:\n" + defs)

#         return (f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
#                 f"<|im_start|>user\n{input_prompt}\n"
#                 )


args = read_args()
defs: FuncDefs = args.defs
prompts: Prompts = args.prompts
model = Small_LLM_Model(args.model, device='cpu')
tokenizer = Tokenizer(model)

str_defs = defs.model_dump_json()
answers: list[dict[Any]] = []
for task in prompts.all:
    prompt_ids: list[int] = model.encode(system_prompt)[0].tolist()
    answer = generate_record(
        get_logits_fn=model.get_logits_from_input_ids,
        # encode_fn=lambda x: model.encode(x)[0].tolist(),
        encode_fn=tokenizer.encode,
        functions=defs.all,
        base_ids=prompt_ids,
        tokenizer=tokenizer,
        prompt=system_prompt,
        task=task
    )
    print(answer)
    answers.append(answer)
with open('data/output/answers.json', 'w') as f:
    json.dump(answers, f)
# model = Small_LLM_Model()
# vocab = Tokenizer(model)
# encoded = model.encode(text)[0].tolist()
# my_encoded = vocab.encode(text)
# with open('data/output/encoded.txt', 'w') as f:
#     f.write(str(encoded))
# with open('data/output/my_encoded.txt', 'w') as f:
#     f.write(str(my_encoded))
# with open('data/output/sdk.json', 'w') as f:
#     f.write(model.decode(encoded))
# with open('data/output/my.json', 'w') as f:
#     f.write(vocab.decode(encoded))
