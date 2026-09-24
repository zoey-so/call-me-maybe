import json
from dataclasses import dataclass
from typing import Any, Protocol
from collections.abc import Callable

from .tokenizer import VocabIndex
from .models import FuncDef, Prompts, FuncDefs, ParamType
import time

TIME = 0  # just for tests


class LLMSdk(Protocol):
    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...
    from torch import Tensor
    def decode(self, ids: list[int]) -> str: ...
    def encode(self, text: str) -> Tensor: ...


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
    node: "_TrieNode", vocab: VocabIndex
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
        # print("allowed len: ", len(vocab.ids_starting_with_chars(edge_char)))
        for token_id in vocab.ids_starting_with_chars(edge_char):
            text = vocab.id_to_byte_txt[token_id]
            walk_node = node
            ok = True
            for ch in text:
                next_node = walk_node.children.get(ch)
                if next_node is None:
                    ok = False
                    break
                walk_node = next_node
            if ok:
                # print("OK", token_id, vocab.id_to_byte_txt[token_id])
                candidates[token_id] = walk_node
    return candidates


def generate_from_closed_set(
    sdk: LLMSdk, context_ids: list[int], vocab: VocabIndex, options: list[str]
) -> tuple[str, list[int]]:
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
        while len(node.children) == 1:
            auto_completed = True
            (char, next_node), = node.children.items()
            auto_completed_text += char
            prev_node = node
            node = next_node
        if auto_completed:
            if node.is_end:
                generated_ids.extend(vocab.encode(auto_completed_text))
                break
            generated_ids.extend(vocab.encode(auto_completed_text[:-1]))
            node = prev_node
            auto_completed_text = ""
        st = time.time()
        candidates = _allowed_from_trie_node(node, vocab)
        et = time.time()
        print(f"allowed time = {(et-st):.3f}")
        if not candidates:
            raise RuntimeError(
                "Must be some error in generate_from_closed_set: "
                "no valid continuation from the current trie "
                f"state (options={options!r}, generated so far={generated_ids!r}). "
            )

        current_ids = context_ids + generated_ids
        logits = sdk.get_logits_from_input_ids(current_ids)
        best_logit = max(range(len(logits)), key=lambda i: logits[i])
        if best_logit == 151645:
            break
        # print("best_logit:", best_logit, vocab.id_to_byte_txt.get(best_logit, "Not found"))
        best_id = max(candidates, key=lambda i: logits[i])
        generated_ids.append(best_id)
        node = candidates[best_id]
    # print(sdk.decode(generated_ids))
    return sdk.decode(generated_ids), generated_ids


def generate_boolean(
    sdk: LLMSdk, context_ids: list[int], vocab: VocabIndex
) -> tuple[bool, list[int]]:
    """Calling generate_from_closed_set with true and false words.
    """
    matched, ids = generate_from_closed_set(sdk, context_ids, vocab, ["true", "false"])
    return matched == "true", ids


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
    sdk: LLMSdk, context_ids: list[int], vocab: VocabIndex, max_tokens: int = 16
) -> tuple[float, list[int]]:
    """Constrained generatung a JSON number (int or float)
    Using DFA to handle the leading - and floating point numbers.
    We need to have at least one digit. After that if anything other than
    digit appears its treated as end of number and we return.
    """
    generated_ids: list[int] = []
    state = "START"

    for _ in range(max_tokens):
        start_chars = _NUMBER_STATE_START_CHARS[state]
        end_states: dict[int, str] = {}
        for token_id in vocab.ids_starting_with_chars(start_chars):
            end_state = _number_token_end_state(vocab.id_to_byte_txt[token_id], state)
            if end_state is not None:
                end_states[token_id] = end_state

        if not end_states:
            if state in _NUMBER_ACCEPTING:
                break
            raise RuntimeError(
                f"grammar error: number generation stuck in state {state!r} "
                f"with no valid continuation (generated so far: {generated_ids!r})"
            )

        current_ids = context_ids + generated_ids
        logits = sdk.get_logits_from_input_ids(current_ids)
        unconstrained_top = max(range(len(logits)), key=lambda i: logits[i])

        if state in _NUMBER_ACCEPTING and unconstrained_top not in end_states:
            break

        best_id = max(end_states, key=lambda i: logits[i])
        generated_ids.append(best_id)
        state = end_states[best_id]

    text = "".join(vocab.id_to_byte_txt[i] for i in generated_ids)
    if not text or text == "-":
        raise RuntimeError(
            f"number generation produced no usable digits (text={text!r}) "
            "-- max_tokens may be too low, or the model never found the "
            "number-start characters worth generating for this prompt"
        )
    return float(text), generated_ids


def _is_valid_string_content(text: str) -> bool:
    disallowed_string_chars = {'"', "\\"}
    if not text:
        return False
    for ch in text:
        if ch in disallowed_string_chars or ord(ch) < 0x20:
            return False
    return True


def generate_string(
    sdk: LLMSdk, context_ids: list[int], vocab: VocabIndex, max_tokens: int = 40
) -> tuple[str, list[int]]:
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
        current_ids = context_ids + generated_ids
        is_valid = False
        logits = sdk.get_logits_from_input_ids(current_ids)
        while not is_valid:
            top_id = max(range(len(logits)), key=lambda i: logits[i])
            txt = vocab.id_to_byte_txt.get(top_id, "")
            is_valid = _is_valid_string_content(txt)
            # print(f"top id so far: {top_id!r}, {txt!r}, {is_valid}")
            tokens_ids = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
            # next_token_id = max(range(0, len(logits)), key=lambda i: logits[i])
            print("Top 3: ", f"{[sdk.decode(v[0]) for v in tokens_ids[:4]]}")
            if top_id == 151645:
                is_end = True
                break
            # if is_valid:
            #     is_anything = True
            # elif is_anything:
            #     is_end = True
            #     break
            elif '"' in vocab.id_to_byte_txt.get(top_id):
                generated_ids += vocab.encode(vocab.id_to_byte_txt.get(top_id).split('"')[0])
                is_end = True
                break
            if not is_valid:
                logits[top_id] = float('-inf')
        if is_end:
            break
        generated_ids.append(top_id)
    # print("Generated in string: ", vocab.decode(generated_ids))
    text = sdk.decode(generated_ids) if generated_ids else ""
    # print("Generated in string: ", text)
    return text, generated_ids


def generate_parameter_value(
    sdk: LLMSdk, context_ids: list[int], vocab: VocabIndex, param_type: str
) -> tuple[Any, list[int]]:
    if param_type == "number":
        return generate_number(sdk, context_ids, vocab)
    if param_type == "string":
        return generate_string(sdk, context_ids, vocab)
    if param_type == "boolean":
        return generate_boolean(sdk, context_ids, vocab)
    raise ValueError(
        f"unsupported parameter type {param_type!r}"
    )
    #  TODO: arrays and lists


def select_function(
    sdk: LLMSdk, context_ids: list[int], vocab: VocabIndex, functions: list[FuncDef]
) -> tuple[FuncDef, list[int]]:
    """Calls generate_from_closed_set()
    with function names to get function name.
    """
    names = [f.name for f in functions]
    # print("names: ", names)
    matched_name, ids = generate_from_closed_set(sdk, context_ids, vocab, names)
    return next(f for f in functions if f.name == matched_name), ids


def generate_record(
    sdk: LLMSdk,
    encode_fn: Callable[[str], list[int]],
    base_ids: list[int],
    vocab: VocabIndex,
    prompt: str,
    functions: list[FuncDef],
    task: str
) -> dict[str, Any]:
    """Generation of name and parameters for a single prompt.
    Parameters
    ----------
    sdk: LLMSdk Protocol
    encode_fn: Callable[[str], list[int]]
        function for encoding returning list of token ids.
    base_ids: list[int]
    """
    global TIME
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
    # print(sdk.decode(funcs_ids))
    chosen, name_ids = select_function(sdk, funcs_ids, vocab, functions)
    prompt += (f"Allowed functions are: {chosen.model_dump_json()}<|im_end|>\n"
               )
    # prompt += (f"Allowed functions are: {chosen.model_dump_json()}\n"
    #            )
    prompt += user_input
    st = time.time()
    running_ids = encode_fn(prompt)
    et = time.time()
    TIME += et - st
    answer += f'{chosen.name}",' + '"arguments":{"name": '
    # running_ids += encode_fn(answer)

    parameters: dict[str, dict[str, ParamType]] = {}
    for i, (param_name, param_type) in enumerate(chosen.parameters.items()):
        if i > 0:
            answer += ','
        answer += f'"{param_name}": '
        if param_type["type"] == ParamType.STR:
            answer += '"'
        st = time.time()
        param_prompt = encode_fn(answer)
        et = time.time()
        TIME += et - st
        new_prompt = running_ids + param_prompt
        # print(sdk.decode(new_prompt))
        value, value_ids = generate_parameter_value(sdk, new_prompt, vocab, param_type["type"].value)
        answer += str(value)
        if param_type["type"] == ParamType.STR:
            answer += '"'
        parameters[param_name] = value
    # for i, (param_name, param_type) in enumerate(chosen.parameters.items()):
        # if i > 0:
        #     running_ids += encode_fn(",")
        # param_prompt = encode_fn(f"Give me only the paramter {param_name} value<im_end>\n"
        #                             "<|im_start|>assistant\n")
        # new_prompt = running_ids + param_prompt
        # print(sdk.decode(new_prompt))
        # value, value_ids = generate_parameter_value(sdk, new_prompt, vocab, param_type["type"].value)
        # running_ids += encode_fn(f"Parameter {param_name} value is ")
        # running_ids += value_ids
        # running_ids.append(vocab.uni_txt_to_id[vocab.byte_to_unicode[ord('\n')]])
        # parameters[param_name] = value
    tet = time.time()
    print(f"total time: {tet-tst:.3f}")
    print(f"time in encode: {TIME:.3f}")
    return {"prompt": task, "name": chosen.name, "parameters": parameters}
