import json
from typing import Any, Protocol
from collections.abc import Callable

from .tokenizer import Tokenizer
from .models import FuncDef, Prompts, FuncDefs, ParamType
from .vis import TraceStep
import time

TIME = 0  # just for tests


class _TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict[str, "_TrieNode"] = {}
        self.is_end = False


def _build_trie(strings: list[str]) -> _TrieNode:
    root = _TrieNode()
    for s in strings:
        node = root
        for ch in s:
            node = node.children.setdefault(ch, _TrieNode())
        node.is_end = True
    return root


def _allowed_from_trie_node(
    node: "_TrieNode", tokenizer: Tokenizer
) -> dict[int, "_TrieNode"]:
    """Masking mechanism:
    given a char by char trie, get all token ids starting with that char
    check if other chars in token dont drop off the trie
    Returns
    --------
    dict[int, "_TrieNode"]
    with all possible token_ids as keys
    and the node we land on if we chose the token as values
    """
    candidates: dict[int, "_TrieNode"] = {}
    for edge_char in node.children:
        # print("allowed len: ", len(tokenizer.ids_starting_with_chars(edge_char)))
        for token_id in tokenizer.ids_starting_with_chars(edge_char):
            text = tokenizer.id_to_byte_txt[token_id]
            walk_node = node
            ok = True
            for ch in text:
                next_node = walk_node.children.get(ch)
                if next_node is None:
                    ok = False
                    break
                walk_node = next_node
            if ok:
                # print("OK", token_id, tokenizer.id_to_byte_txt[token_id])
                candidates[token_id] = walk_node
    return candidates


def generate_from_closed_set(
    get_logits_fn: Callable[[list[int]], list[float]],
    context_ids: list[int],
    tokenizer: Tokenizer,
    options: list[str],
    steps: list[TraceStep] | None = None,
    stage: str = 'closed_set',
    prompt: str = '',
    _id: str = '',
    answer: str | None = None,
    step_index: int = 0
) -> tuple[str, list[int], int]:
    """Constrained generating from set of strings.
    Builds tries char by char for every option,
    masks tokens that can't be any of the option.
    Chooses safely the best after masking.
    Using trieing if there is only one char possible to autocomplete
    and save unneeded calls.
    """
    trie_root = _build_trie(options)
    generated_ids: list[int] = []
    auto_completed_text: str = ""
    node = trie_root
    # print("node.children:", node.children)

    while not node.is_end:
        auto_completed = False
        st = time.time()
        while len(node.children) == 1:
            auto_completed = True
            (char, next_node), = node.children.items()
            auto_completed_text += char
            prev_node = node
            node = next_node
        if auto_completed:
            et = time.time()
            answer += auto_completed_text
            if steps is not None:
                steps.append(
                    TraceStep(
                        stage=stage,
                        step_index=step_index,
                        forced=True,
                        chosen_id=1,
                        chosen_text=auto_completed_text,
                        prompt=prompt,
                        _id=_id,
                        _time=f"{et - st:.3f}",
                        answer=answer
                    )
                )
                step_index += 1
            if node.is_end:
                generated_ids.extend(tokenizer.encode(auto_completed_text))
                break
            generated_ids.extend(tokenizer.encode(auto_completed_text[:-1]))
            answer = answer[:-1]
            node = prev_node
            auto_completed_text = ""
        candidates = _allowed_from_trie_node(node, tokenizer)
        if not candidates:
            raise RuntimeError(
                "Must be some error in generate_from_closed_set: "
                "no valid continuation from the current trie "
                f"state (options={options!r}, generated so far={generated_ids!r}). "
            )

        current_ids = context_ids + generated_ids
        st = time.time()
        logits = get_logits_fn(current_ids)
        best_logit = max(range(len(logits)), key=lambda i: logits[i])
        if best_logit == 151645:
            break
        # print("best_logit:", best_logit, tokenizer.id_to_byte_txt.get(best_logit, "Not found"))
        best_id = max(candidates, key=lambda i: logits[i])
        et = time.time()
        generated_ids.append(best_id)
        node = candidates[best_id]
        answer += tokenizer.id_to_byte_txt[best_id]
        if steps is not None:
            steps.append(
                TraceStep(
                    stage=stage,
                    step_index=step_index,
                    forced=False,
                    chosen_id=1,
                    chosen_text=tokenizer.id_to_byte_txt[best_id],
                    prompt=prompt,
                    _id=_id,
                    _time=f"{et - st:.3f}",
                    answer=answer
                )
            )
            step_index += 1
    # print(tokenizer.decode(generated_ids))
    return tokenizer.decode(generated_ids), generated_ids, step_index


def generate_boolean(
    get_logits_fn: Callable[[list[int]], list[float]],
    context_ids: list[int],
    tokenizer: Tokenizer,
    steps: list[TraceStep] | None = None,
    _id: str = '',
    answer: str | None = None,
    step_index: int = 0
) -> tuple[str, list[int], int]:
    """Calling generate_from_closed_set with true and false words.
    """
    matched, ids, step_index = generate_from_closed_set(
        get_logits_fn, context_ids, tokenizer, ["true", "false"],
        steps=steps, stage="boolean", prompt="",
        _id=_id, answer=answer, step_index=step_index)
    return matched, ids, step_index


_NUMBER_TRANSITIONS: dict[str, dict[str, str]] = {
    "START": {"-": "SIGN_SEEN", **{d: "INT_DIGITS" for d in "0123456789"}},
    "SIGN_SEEN": {d: "INT_DIGITS" for d in "0123456789"},
    "INT_DIGITS": {".": "DOT_SEEN", **{d: "INT_DIGITS" for d in "0123456789"}},
    "DOT_SEEN": {d: "FRAC_DIGITS" for d in "0123456789"},
    "FRAC_DIGITS": {d: "FRAC_DIGITS" for d in "0123456789"},
}
_NUMBER_ACCEPTING = {"INT_DIGITS", "FRAC_DIGITS"}
_NUMBER_STATE_START_CHARS: dict[str, str] = {
    "START": "-0123456789",
    "SIGN_SEEN": "0123456789",
    "INT_DIGITS": "0123456789.",
    "DOT_SEEN": "0123456789",
    "FRAC_DIGITS": "0123456789",
}


def _number_token_end_state(text: str, start_state: str) -> str | None:
    """Char by char trie of states in number. From start state to current."""
    state = start_state
    for ch in text:
        next_state = _NUMBER_TRANSITIONS.get(state, {}).get(ch)
        if next_state is None:
            return None
        state = next_state
    return state


def generate_number(
    get_logits_fn: Callable[[list[int]], list[float]],
    context_ids: list[int],
    tokenizer: Tokenizer,
    steps: list[TraceStep] | None = None,
    _id: str = '',
    answer: str | None = None,
    step_index: int = 0,
    max_tokens: int = 16
) -> tuple[float | int, list[int], int]:
    """Constrained generatung a JSON number (int or float)
    Using DFA to handle the leading - and floating point numbers.
    We need to have at least one digit. After that if anything other than
    digit appears its treated as end of number and we return.
    """
    generated_ids: list[int] = []
    state = "START"

    for _ in range(max_tokens):
        st = time.time()
        start_chars = _NUMBER_STATE_START_CHARS[state]
        end_states: dict[int, str] = {}
        for token_id in tokenizer.ids_starting_with_chars(start_chars):
            end_state = _number_token_end_state(
                tokenizer.id_to_byte_txt[token_id], state)
            if end_state is not None:
                end_states[token_id] = end_state

        if not end_states:
            if state in _NUMBER_ACCEPTING:
                break
            raise RuntimeError(
                f"grammar error: number generation stuck in state {state!r} "
                "with no valid continuation "
                f"(generated so far: {generated_ids!r})"
            )

        current_ids = context_ids + generated_ids
        logits = get_logits_fn(current_ids)
        unconstrained_top = max(range(len(logits)), key=lambda i: logits[i])
        print("top id: ",
              tokenizer.id_to_byte_txt.get(unconstrained_top, "Not found"))
        if state in _NUMBER_ACCEPTING and unconstrained_top not in end_states:
            break
        best_id = max(end_states, key=lambda i: logits[i])
        generated_ids.append(best_id)
        state = end_states[best_id]
        answer += tokenizer.id_to_byte_txt.get(best_id, "")
        if steps is not None:
            et = time.time()
            steps.append(
                TraceStep(
                    stage="generate number",
                    step_index=step_index,
                    forced=False,
                    chosen_id=best_id,
                    chosen_text=tokenizer.id_to_byte_txt.get(best_id, ""),
                    prompt="",
                    _id=_id,
                    _time=f"{et - st:.3f}",
                    answer=answer
                )
            )
            step_index += 1
    text = "".join(tokenizer.id_to_byte_txt[i] for i in generated_ids)
    if not text or text == "-":
        raise RuntimeError(
            f"number generation produced no usable digits (text={text!r}) ")
    res = float(text) if "." in text else int(text)
    return res, generated_ids, step_index


def _is_valid_string_content(text: str) -> bool:
    disallowed_string_chars = {""}
    if not text:
        return False
    for ch in text:
        if ch in disallowed_string_chars or ord(ch) < 0x20:
            return False
    return True


def generate_string(
    get_logits_fn: Callable[[list[int]], list[float]],
    context_ids: list[int],
    tokenizer: Tokenizer,
    steps: list[TraceStep] | None = None,
    _id: str = '',
    answer: str | None = None,
    step_index: int = 0,
    max_tokens: int = 40
) -> tuple[str, list[int], int]:
    """Most open generaation type for strings.
    No masking because to many possibilities,
    just checking the tokens for quotes and escape chars.
    Asumming a quote is ending quote - extracting anything before quote
    that can be still part of the answer.
    """
    generated_ids: list[int] = []
    # is_anything = False
    is_end = False
    for _ in range(max_tokens):
        st = time.time()
        current_ids = context_ids + generated_ids
        is_valid = False
        logits = get_logits_fn(current_ids)
        while not is_valid:
            top_id = max(range(len(logits)), key=lambda i: logits[i])
            txt = tokenizer.id_to_byte_txt.get(top_id, "")
            is_valid = _is_valid_string_content(txt)
            # print(f"top id so far: {top_id!r}, {txt!r}, {is_valid}")
            tokens_ids = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
            # next_token_id = max(range(0, len(logits)), key=lambda i: logits[i])
            print("Top 3: ", f"{[tokenizer.decode([v[0]]) for v in tokens_ids[:4]]}")
            if top_id == 151645:
                is_end = True
                break
            # if is_valid:
            #     is_anything = True
            # elif is_anything:
            #     is_end = True
            #     break
            elif '"' in txt:
                if generated_ids or txt.count('"') > 1:
                    is_end = True
                    break
            if not is_valid or (not generated_ids and '"' not in txt):
                logits[top_id] = float('-inf')
        generated_ids.append(top_id)
        et = time.time()
        if steps is not None:
            answer += txt
            steps.append(
                TraceStep(
                    stage="generate string",
                    step_index=step_index,
                    forced=False,
                    chosen_id=top_id,
                    chosen_text=txt,
                    prompt="",
                    _id=_id,
                    _time=f"{et - st:.3f}",
                    answer=answer
                )
            )
            step_index += 1
        if is_end:
            break
        print("generated_ids: ", generated_ids, tokenizer.decode(generated_ids))
    # print("Generated in string: ", tokenizer.decode(generated_ids))
    text = tokenizer.decode(generated_ids) if generated_ids else ""
    if '"' in text:
        text = text.split('"')[1]
    # print("Generated in string: ", text)
    return text, generated_ids, step_index


def generate_parameter_value(
    get_logits_fn: Callable[[list[int]], list[float]],
    context_ids: list[int],
    tokenizer: Tokenizer,
    param_type: str,
    steps: list[TraceStep] | None = None,
    _id: str = '',
    answer: str | None = None,
    step_index: int = 0
) -> tuple[Any, list[int], int]:
    if param_type == "number":
        return generate_number(get_logits_fn, context_ids, tokenizer, steps, _id, answer, step_index)
    if param_type == "string":
        return generate_string(get_logits_fn, context_ids, tokenizer, steps, _id, answer, step_index)
    if param_type == "boolean":
        return generate_boolean(get_logits_fn, context_ids, tokenizer, steps, _id, answer, step_index)
    raise ValueError(
        f"unsupported parameter type {param_type!r}"
    )
    #  TODO: arrays and lists


def select_function(
    get_logits_fn: Callable[[list[int]], list[float]],
    context_ids: list[int],
    tokenizer: Tokenizer,
    functions: list[FuncDef],
    steps: list[TraceStep] | None = None,
    prompt: str = '',
    _id: str = '',
    answer: str | None = None,
    step_index: int = 0
) -> tuple[FuncDef, list[int]]:
    """Calls generate_from_closed_set()
    with function names to get function name.
    """
    names = [f.name for f in functions]
    # print("names: ", names)
    matched_name, ids, step_index = generate_from_closed_set(
        get_logits_fn, context_ids, tokenizer,
        names, steps, "select_function", prompt, _id, answer, step_index)
    return next(
        f for f in functions if f.name == matched_name), ids, step_index


def generate_record(
    get_logits_fn: Callable[[list[int]], list[float]],
    encode_fn: Callable[[str], list[int]],
    base_ids: list[int],
    tokenizer: Tokenizer,
    prompt: str,
    functions: list[FuncDef],
    task: str,
    steps: list[TraceStep] | None = None,
    _id: str = ''
) -> dict[str, Any]:
    """Generation of name and parameters for a single prompt.
    Parameters
    ----------
    get_logits_fn: Callable[[list[int]], list[float]]
        function for getting llm output for given token ids.
    encode_fn: Callable[[str], list[int]]
        function for encoding returning list of token ids.
    base_ids: list[int]
    """
    global TIME
    step_index = 0
    running_ids = list(base_ids)
    tst = time.time()
    funcs_prompt = ("Allowed functions are: "
                    f"{[f.model_dump_json() for f in functions]}<|im_end|>\n"
                    )
    # # funcs_prompt = ("Allowed functions are: "
    #                 f"{[f.model_dump_json() for f in functions]}\n"
    #                 )
    user_input = f"<|im_start|>user\n{task}\n<|im_end|>\n"
    # user_input = f"The prompt is: {task}\n"
    answer = ("<|im_start|>assistant\n"
              '{"function": "')
    # answer = ('Valid JSON is: {"function": "')
    st = time.time()
    funcs_ids = encode_fn(prompt + funcs_prompt + user_input + answer)
    et = time.time()
    TIME += et - st
    # print(tokenizer.decode(funcs_ids))
    chosen, ids, step_index = select_function(
        get_logits_fn, funcs_ids, tokenizer,
        functions, steps, task, _id, answer[21:], step_index)
    prompt += (f"Allowed functions are: {chosen.model_dump_json()}<|im_end|>\n"
               )
    # prompt += (f"Allowed functions are: {chosen.model_dump_json()}\n"
    #            )
    prompt += user_input
    st = time.time()
    running_ids = encode_fn(prompt)
    et = time.time()
    TIME += et - st
    answer += f'{chosen.name}", ' + '"arguments":{'
    # running_ids += encode_fn(answer)

    parameters: dict[str, dict[str, ParamType]] = {}
    for i, (param_name, param_type) in enumerate(chosen.parameters.items()):
        if i > 0:
            answer += ', '
        answer += f'"{param_name}":'
        # if param_type["type"] == ParamType.STR:
        #     answer += '"'
        st = time.time()
        param_prompt = encode_fn(answer)
        et = time.time()
        TIME += et - st
        new_prompt = running_ids + param_prompt
        print("answer: ", answer)
        # print(tokenizer.decode(new_prompt))
        value, value_ids, step_index = generate_parameter_value(
            get_logits_fn, new_prompt,
            tokenizer, param_type["type"].value, steps, _id, answer[21:], step_index)
        answer += f' "{str(value)}"' if param_type["type"] == ParamType.STR else f' {str(value)}'
        parameters[param_name] = value
    # for i, (param_name, param_type) in enumerate(chosen.parameters.items()):
        # if i > 0:
        #     running_ids += encode_fn(",")
        # param_prompt = encode_fn(f"Give me only the paramter {param_name} value<im_end>\n"
        #                             "<|im_start|>assistant\n")
        # new_prompt = running_ids + param_prompt
        # print(tokenizer.decode(new_prompt))
        # value, value_ids = generate_parameter_value(get_logits_fn, new_prompt, tokenizer, param_type["type"].value)
        # running_ids += encode_fn(f"Parameter {param_name} value is ")
        # running_ids += value_ids
        # running_ids.append(tokenizer.uni_txt_to_id[tokenizer.byte_to_unicode[ord('\n')]])
        # parameters[param_name] = value
    tet = time.time()
    print(f"total time: {tet-tst:.3f}")
    print(f"time in encode: {TIME:.3f}")
    if steps is not None:
        steps.append(
            TraceStep(
                stage="end",
                step_index=step_index,
                forced=False,
                chosen_id=1,
                chosen_text=answer[21:],
                prompt=prompt,
                _id=_id,
                _time=f"{tet - tst:.3f}",
                answer=answer[21:] + '}'
            )
        )
    return {"prompt": task, "name": chosen.name, "parameters": parameters}
