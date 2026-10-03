*This project has been created as part of the 42 curriculum by smilch.*

# call me maybe

## Description

**call me maybe** is a function-calling tool for small language models: given a
natural-language prompt and a set of available function definitions, it
produces a structured, schema-valid record of which function should be
called and with what arguments — without ever relying on the model to
"just produce correct JSON" on its own.

The core problem this project solves: small models (this project targets
`Qwen/Qwen3-0.6B` by default) are unreliable at producing well-formed,
schema-compliant JSON through prompting alone. Instead of hoping the model
gets the format right, this project uses **constrained decoding** —
masking the model's next-token logits at every generation step so that
only tokens consistent with the required JSON Schema can ever be selected.
The output is therefore *guaranteed* to be valid JSON matching the given
function definitions, while the *content* (which function, which argument
values) is still driven by the model's own predictions.

Beyond the mandatory scope, this project also implements:
- A from-scratch byte-level BPE tokenizer (`Tokenizer`, in `tokenizer.py`),
  built directly from a model's `tokenizer.json` — no dependency on
  `llm_sdk`'s own `encode`/`decode`.
- Support for **nested parameter types** — `array` (homogeneous, via a
  JSON-Schema-style `items` field) and `object` (via a `properties` field)
  — recursively, so a parameter can be an array of objects, an object
  containing arrays, and so on.
- A **live, browser-based visualization** of the generation process, served
  over Server-Sent Events from a small stdlib-only HTTP server — watch each
  constrained-decoding decision as it happens, no external packages needed.
- Portability beyond the default model: this implementation has also been
  tested against models from the **HuggingFace SmolLM2** family.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
uv sync
```

This installs the project's dependencies, including `pydantic` and
`regex` (see **Concerns**, below, about the latter), plus whatever
`llm_sdk` itself requires (PyTorch, Hugging Face `transformers`,
`huggingface_hub`).

### Running the program

```bash
uv run python -m src [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `-fd, --functions_definition` | `data/input/functions_definition.json` | Path to the function definitions file |
| `-i, --input` | `data/input/function_calling_tests.json` | Path to the prompts file |
| `-o, --output` | `data/output/function_calls.json` | Path to write results to |
| `-m, --model` | `Qwen/Qwen3-0.6B` | Hugging Face model id to use |
| `-d, --device` | auto-detect (`cuda` > `mps` > `cpu`) | Force a specific compute device |
| `-v, --verbose` | off | Enable the live browser visualization |
| `-p, --port` | `4242` | Port for the visualization server (with `-v`) |

Both `--functions_definition` and `--input` are validated against a
pydantic schema at load time — a malformed or missing file produces a
clear command-line error rather than a crash.

Example:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition_ext.json \
  --input data/input/function_calling_tests_ext.json \
  --output data/output/function_calls.json
```

### Watching generation live

```bash
uv run python -m src --verbose --port 4242
```

Prints a URL (`http://localhost:4242/`) to open in a browser *before*
generation starts — the program waits for a viewer to connect
(`live.wait_for_viewer()`) so the very first steps aren't missed. Each
prompt gets its own card in the viewer, filling in with every
constrained-decoding decision as it happens, and a running total time at
the end.

## Algorithm explanation

Generation for one prompt (`generate_record`) proceeds in two phases, both
built on the same underlying idea: **the model's logits are only ever
consulted to choose among a set of candidates already known, by
construction, to be valid** — never to decide validity itself.

**Phase 1 — function selection.** `select_function` builds a character-level
trie (`_build_trie`) out of every candidate function name, then walks it
one decision at a time via `generate_from_closed_set`. At each trie node,
`_allowed_from_trie_node` finds every vocabulary token whose full decoded
text stays on a valid path from that node (checked character-by-character,
so a multi-character token is only accepted if *every* character in it
keeps the walk valid) — restricted to candidates that actually start with
one of the node's outgoing edge characters, rather than scanning the whole
vocabulary. The model's logits (`get_logits_fn`) are called only to choose
*among* that candidate set; `argmax` over the masked candidates picks the
next token.

A genuine optimization on top of the base trie walk: whenever the trie has
only one possible next *character* (`len(node.children) == 1`), the walk
advances through as many such forced characters as possible without
calling the model at all, accumulating them as plain text and converting
the whole run to real token ids in one `tokenizer.encode()` call at the
end — rather than spending a model call on every individual forced
character. One subtlety worth being precise about for your own defense:
the walk deliberately backs off by exactly one character
(`node = prev_node`) before handing off to the real masking step, rather
than committing the very last forced character too. This matters because
a *token* in the vocabulary can span more characters than a single trie
edge — backing off one character gives `_allowed_from_trie_node` the
chance to discover a longer token that spans across what looks, at the
character level, like a forced step directly into the branch point,
rather than artificially locking in a shorter token right before a
decision that a longer one could have resolved in one step.

**Phase 2 — parameter generation.** `generate_parameter_value` dispatches
on each parameter's declared type (recursively, for `object`/`array`):

- **`number`** (`generate_number`) — a hand-rolled DFA
  (`_NUMBER_TRANSITIONS`) over states `START → SIGN_SEEN/INT_DIGITS →
  DOT_SEEN → FRAC_DIGITS`, restricting candidate tokens per state to those
  starting with a valid next character (`_NUMBER_STATE_START_CHARS`) and
  walking each candidate's *entire* text through the DFA
  (`_number_token_end_state`) to confirm it stays valid throughout, not
  just its first character. Since the surrounding JSON punctuation isn't
  generated by the model (it's written directly into the `answer`
  scratchpad as literal text), there's no explicit "stop" token in
  context — so once in an accepting state, generation stops the moment the
  model's own *unconstrained* top prediction no longer continues the
  number, treating that as the model's own signal the value is finished.
- **`string`** (`generate_string`) — rather than enumerating valid
  candidates up front (impractical when almost the entire vocabulary is
  potentially valid string content), this checks the model's actual top
  pick at each step and masks it out (`logits[top_id] = -inf`) only if
  it's invalid, repeating until a valid one surfaces. The opening and
  closing quote characters are generated as real tokens (not spliced in
  externally): the very first accepted token must itself contain a quote,
  and a later quote-bearing token (once content exists) is treated as the
  closing quote, after which the surrounding quotes are stripped back out
  (`text.split('"')[1]`) to get the clean value.
- **`boolean`** (`generate_boolean`) — reuses `generate_from_closed_set`
  with the two-element option set `["true", "false"]`, so it automatically
  handles tokenizers that split either word into more than one token.
- **`object`** (inline in `generate_parameter_value`) — recursively
  generates each declared property in turn, writing each one's value back
  into the shared `answer` scratchpad (via `json.dumps(value)`) before
  generating the next property, so later properties' generation sees
  earlier ones already decided.
- **`array`** — repeatedly calls `generate_parameter_value` for
  `param_def.items` (which may itself be an `object` or another `array`,
  giving arbitrary nesting depth) until an element generator returns
  `None`. `generate_number`/`generate_string` return `None` specifically
  when the model's own preference points toward closing the array (text
  containing `]`) rather than continuing — the same mechanism that lets an
  *empty* array terminate immediately, with no special-cased "zero
  elements" branch needed.

Across both phases, the `answer` string is the single source of truth for
"what has been decided so far, rendered as text" — every generated value,
whatever type it is or how many tokens it took, gets folded back into
`answer` via `json.dumps`, so each subsequent decision's model call is
conditioned on the real, literal JSON text of everything already committed
to, not just an abstract record of prior choices.


## Design decisions

- **Nested parameter types via a recursive, discriminated pydantic
  schema.** `models.py` defines `ParamDef = FlatParamDef | ObjParamDef |
  ArrParamDef` as a self-referential union — `ObjParamDef.properties: dict[str,
  ParamDef]` and `ArrParamDef.items: ParamDef` both refer back to the union
  that contains them, resolved via `from __future__ import annotations`
  plus explicit `.model_rebuild()` calls. This lets `functions_definition.json`
  validation reject a malformed nested schema (e.g. an `array` missing
  `items`, or an `object` missing `properties`) before generation ever
  starts, rather than failing deep inside a recursive generator.
- **`None` as a universal "no more elements" sentinel.** Rather than giving
  array generation its own special-cased stopping logic, `generate_number`
  and `generate_string` both already return `None` when the model's
  context signals termination — the array loop just checks `if value is
  None: break`. Elegant, but see **Concerns** for the one sharp edge this
  creates.
- **Context re-synchronization via `json.dumps`, not raw token tracking.**
  When a recursive call (an `object` or `array` element) returns, its
  value is folded back into the parent's `answer` scratchpad as
  `json.dumps(value)` — the *semantic* result, not the specific tokens
  used to produce it. This sidesteps needing to pass token sequences up
  and down the recursion; any value, however it was generated, becomes
  consistent literal text for whatever comes next.
- **Tokenizer built from `tokenizer.json` alone**, not from separate
  `vocab.json`/`merges.txt` calls — `Tokenizer.__init__` takes a single
  `tf_path` and gets vocabulary, merges, and the pre-tokenizer regex all
  from one validated `TokenizerFile` model. `BPEModel.merges`'s
  `field_validator` explicitly normalizes both historical merge-entry
  shapes (`"left right"` strings and `[left, right]` pairs) rather than
  assuming one — directly guarding against the kind of `tokenizers`-library
  version drift this project's own tokenizer.json once hit during
  development (see **Challenges faced**).
- **The `regex` package (not stdlib `re`) for pretokenization**, so the
  real pretokenizer pattern pulled from `tokenizer.json` — which uses
  Unicode property escapes like `\p{L}`/`\p{N}` — can be used verbatim
  rather than approximated with an ASCII-only fallback.
- **Visualization that can't take down generation.** In `__main__.py`,
  starting the viewer server is wrapped in its own `try/except`, falling
  back to `live = None` (visualization silently disabled) rather than
  letting a port conflict or any other startup failure stop the actual
  function-calling run.
- **Server-Sent Events over a stdlib-only HTTP server** (`vis.py`) for the
  live viewer — no extra web framework or websocket library, just
  `http.server.ThreadingHTTPServer` and the browser's native `EventSource`.
  Each browser connection gets its own `queue.Queue`, fed by `VisQueue.append()`
  under a lock, so multiple tabs (or none) can watch the same run safely.

## Performance analysis
The visualisation provides information about time execution for every step, prompt and total time for all prompts.

### Performance for standard prompt file - function_calling_tests.json
valid JSON: 100%

 correct function selection and argument extraction: 100%

Speed (on 42 machine):

- CPU: ~72s


## Challenges faced

- **Terminating open-ended generation (strings, arrays) without an explicit
  stop token in context**, since the surrounding JSON punctuation is
  written as literal text rather than generated. Solved with two related
  techniques: comparing the model's constrained choice against its
  genuine *unconstrained* top pick (numbers stop once the model's real
  preference leaves the valid continuation set), and a `None`-as-sentinel
  convention letting the same per-element generators double as the
  array-termination signal (see **Design decisions**).

## Testing strategy

Two extended fixture files are included: `functions_definition_ext.json`
(14 functions) and `function_calling_tests_ext.json` (38 prompts),
covering:
- All four primitive parameter types (`number`, `string`, `boolean`, and
  `FlatParamDef`'s `null`), including multi-parameter, mixed-type
  functions (`fn_check_number_in_range`, `fn_format_text`)
- Deliberate edge cases: negative numbers, large numbers, decimals, empty
  strings, a negative square root input, informal/non-templated phrasing
  ("What's 7 plus 8?"), and a prompt with no good function match at all
  ("Tell me a joke")
- Nested parameter types at increasing depth: `array[number]`
  (`fn_sum_list`), `object{string, number, boolean}` (`fn_build_profile`),
  and `array[object{number, number}]` (`fn_sum_number_pairs`) — including
  empty-array, single-element, and multi-element cases for the array types


## Example usage

Default run (uses `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`):

```bash
uv run python -m src
```

Run with a specific model and the extended fixtures:

```bash
uv run python -m src \
  -m HuggingFaceTB/SmolLM2-135M \
  -fd data/input/functions_definition_ext.json \
  -i data/input/function_calling_tests_ext.json \
  -o data/output/results.json
```

Run with live visualization on a non-default port:

```bash
uv run python -m src -v -p 5000
```

Force CPU (useful on machines without a supported GPU/MPS backend):

```bash
uv run python -m src -d cpu
```

### Other open-access BPE models to try

This implementation targets `Qwen/Qwen3-0.6B` by default and has also been
tested against the **HuggingFaceTB/SmolLM2** family. Since `Tokenizer`
works from any model's `tokenizer.json` (requiring `model.type == "BPE"`),
any byte-level BPE model with a `tokenizer.json` on the Hub should work in
principle — **though see the hardcoded special-token id noted in
Concerns**, which currently ties correct string-termination behavior
specifically to Qwen's vocabulary numbering. A few genuinely open (no
access request/gating), small-enough-for-local-testing options, confirmed
BPE-tokenizer-based:

| Model | Size | Notes |
|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` / `-360M` | 135M / 360M | Already confirmed working |
| `openai-community/gpt2` | 124M | The original byte-level BPE tokenizer this whole scheme is based on |
| `EleutherAI/pythia-70m` / `-160m` / `-410m` | 70M–410M | Apache 2.0, GPT-NeoX-20B tokenizer (BPE), very fast to download and run |
| `EleutherAI/gpt-neo-125m` | 125M | Open, BPE |
| `Qwen/Qwen2.5-0.5B` | 0.5B | Same tokenizer family as Qwen3 — a good same-family cross-check |

## Resources

- Hugging Face's from-scratch BPE tokenization walkthrough:
  <https://huggingface.co/learn/llm-course/en/chapter6/5>
- Hugging Face's comparison of tokenizer families (BPE / Unigram / WordPiece
  / SentencePiece): <https://huggingface.co/docs/transformers/en/tokenizer_summary>
- Sennrich, Haddow & Birch, *"Neural Machine Translation of Rare Words with
  Subword Units"* — the original BPE-for-NLP paper:
  <https://arxiv.org/abs/1508.07909>
- `llama.cpp`'s GBNF grammar format, a production example of
  constrained/grammar-guided decoding:
  <https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md>
- `Outlines`, a structured-generation library built on the same core idea
  (logit masking against a schema): <https://github.com/dottxt-ai/outlines>
- https://arxiv.org/abs/2310.07075
- https://blog.gopenai.com/byte-level-byte-pair-encoding-bbpe-in-modern-llms-e85695b90685
- https://medium.com/@SuriNaren/the-hidden-engine-of-ai-cracking-the-gpt-tokenizer-9c40129ffcf0
- https://docs.python.org/3/library/http.server.html
- https://json-schema.org/

### AI usage

AI assistance (Claude) was used during this project for:
- Drafting this README
- Code review

The core implementation — `generate.py`, `models.py`, `parser.py`,
`tokenizer.py`, `vis.py`, `viewer.html`, and `__main__.py` — was written
independently; AI assistance did not write these files directly.

