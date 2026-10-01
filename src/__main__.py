from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import Any
import json
import hashlib
import sys

from .tokenizer import Tokenizer
from .parser import read_args
from .models import Prompts, FuncDefs
from .generate import generate_record
from .vis import VisQueue, TraceStep, start_trace_server


SYSTEM_PROMPT = ("<|im_start|>system\nYou are a helpful assistant that "
                 "will perform function calling.\n"
                 "Your job is to first understand what functions you can use, "
                 "and then to select a correct one based on prompt. "
                 "You need to provide a valid JSON, in this format: "
                 '{"function": "function name", '
                 '"arguments":{"name": "value"}}\n'
                 "Remember to put in arguments correctly. "
                 "Pay attention to regex.\n"
                 )


def main() -> None:
    args = read_args()
    live: VisQueue | None = None
    if args.verbose:
        try:
            live = VisQueue()
            server = start_trace_server(live, port=args.port)
            print(
                f"Viewer is running at http://localhost:{server.server_port}/")
            live.wait_for_viewer()
        except Exception as e:
            live = None
            print("Problem occured with starting the server. No visualisation."
                  f" - {e}")

    defs: FuncDefs = args.defs
    prompts: Prompts = args.prompts
    try:
        model = Small_LLM_Model(args.model, device=args.device)
    except Exception as e:
        print(f"Problem occured with initialising the model {args.model} -"
              f" {e}")
        sys.exit(1)
    try:
        tokenizer = Tokenizer(model.get_path_to_tokenizer_file())
    except Exception as e:
        print(f"Problem occured with tokenizer.json file from model -"
              f" {e}")
        sys.exit(1)

    answers: list[dict[str, Any]] = []
    total_time: float = 0.0
    for task in prompts.all:
        try:
            answer, pt = generate_record(
                get_logits_fn=model.get_logits_from_input_ids,
                encode_fn=tokenizer.encode,
                functions=defs.all,
                tokenizer=tokenizer,
                prompt=SYSTEM_PROMPT,
                task=task,
                steps=live,
                _id=hashlib.md5(task.encode()).hexdigest()
            )
        except Exception as e:
            print(f"Problem occured with prompt: {task} - "
                  f"{e}\nSkipping this prompt...")

        answers.append(answer)
        total_time += pt
    try:
        with open(args.output, 'w') as f:
            json.dump(answers, f)
    except Exception as e:
        print("Problem occured with writing the file. "
              "This really shouldn't have happened."
              f"\n{e}")
    if live is not None:
        live.append(TraceStep(
            stage="finished",
            _time=f"{int(total_time // 60)}min {int(total_time % 60)}s"
        ))
        live.finish()


try:
    main()
except KeyboardInterrupt:
    print("Bye, bye!")
    sys.exit(1)

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
