# mt-game: talking through a machine translator

A two-player chat game for studying how machine translation (MT) affects
collaboration. Two people who speak different languages play the **tangram
figure task** (Clark & Wilkes-Gibbs 1986; Yamashita & Ishida 2006). Every
message goes through an MT model in the middle, and the game can show each
side different explanations of the translation.

## The game

- Both players see the same 12 tangram figures.
- The **director** sees them in a target order (slots 1–12) and describes
  them in the chat.
- The **matcher** sees them shuffled, clicks them into slots 1–12, and
  presses **Submit round** when all 12 are placed. Clicking a placed
  figure takes it back out.
- After each submit the same figures come back in a new order, with no
  round limit, so the pair can build up shared names ("the kneeling man").
  Each round's score shows after it.
- Each player types in their own language. The receiver sees only the
  translation.

In the default setup the receiver also sees the **translator's reasoning**
(chain of thought) under each message. Other explanations can be switched
on; see [Settings](#settings).

## Running it

On boa (needs one free GPU, ~16 GB, for the Seed-X translator):

```
cd mt-game
uv sync                        # once: install dependencies
uv run python tangrams.py      # once: download the tangram images
nvidia-smi                     # pick a free GPU
GPU=1 ./play.sh
```

`play.sh` prints the links. On the WireGuard network boa is `192.168.50.7`:

| page | link |
|---|---|
| **developer view** (both players on one screen) | http://192.168.50.7:8000/dev |
| director | http://192.168.50.7:8000/?role=director&lang=en |
| matcher | http://192.168.50.7:8000/?role=matcher&lang=zh |

`lang=` is the player's language: `en zh es fr de ja ko`. The model takes
about 10 s to load once it is cached (the first run downloads ~15 GB).

## Developer view

http://192.168.50.7:8000/dev puts both players side by side, so one person can
play both roles to try things out. It also shows:

- **Board check:** the target order next to the matcher's current
  placements, green where a slot is right and red where it is wrong.
- **Message pipeline:** for every message, what the sender typed, what the
  MT model received (highlighted if a deliberate error was added), what the
  receiver saw, any explanations, and how long it took.
- **Controls:** language dropdowns for each side (then **Load**), the
  tangram set (`classic` or `abstract`; picking one starts a new game), and
  **New game**.
- **Translation relay:** add, reorder, or remove intermediate languages,
  then **Apply relay**. Each message shows its complete route and every
  step's input, output, and translation time.

The developer view shows the answer, so never give it to participants.

## Multiple translation steps

Each message can pass through a sequence of languages before reaching its
receiver, like a multilingual telephone game. Start a three-step relay with:

```
GPU=1 ./play.sh --via es,fr
```

With an English director and Chinese matcher, this produces:

```
Director: English → Spanish → French → Chinese
Matcher:  Chinese → French → Spanish → English
```

Every step translates the previous step's output. The receiver sees the
final result in their own language; intermediate text is available in the
developer view and logs. While a message is processing, players see the
current step and total number of translation steps.

Use any of `en zh es fr de ja ko` as intermediate languages, up to eight
per relay (up to nine translation steps). The sender and receiver languages
are supplied automatically. Adjacent repeated languages are skipped;
non-adjacent repeats are kept, so `en → fr → en → zh` is valid. Two players
can also use the same language and pass messages through other languages
before returning to it.

The developer view's **Translation relay** controls apply changes to future
messages without clearing chat or scores. Existing messages retain their
original routes. The relay persists across game rounds and **New game**.
Use **Use direct translation**, then **Apply relay**, for the original
single-step baseline. Omitting `--via` also starts with direct translation.
Translation steps and tangram game rounds are independent.

Both players must join before messages can be sent. Route/language changes,
round submission, and game resets wait until in-flight messages finish.
If a translation fails, no partial translation is delivered, and the sender
can retry their restored message. Longer chains require more model calls.

Explanations with a relay:

- **Confidence, alternatives, and the translator's explanation** describe
  the final step, using its actual input and output. They do not measure
  accuracy or uncertainty across the full chain.
- **Back-translation** translates the final received text directly into
  the original sender's language in one additional step.
- **History** still compares the sender's original MT input with earlier
  messages and uses a direct translation for repeated-term glosses.
- Deliberate errors are injected once, before the first translation step.

### Local preview without a GPU

The mock translator adds language tags; it does **not** translate the text.
To check the controls without installing model dependencies, use Python 3.12+:

```
cd mt-game
python3 -m venv .venv
.venv/bin/python -m pip install 'flask>=3.1.3'
.venv/bin/python tangrams.py
.venv/bin/python server.py --model mock --via es,fr --explain cot,backtranslation
```

Open http://localhost:8000/dev. The relay wraps all existing model backends;
use the GPU setup above for real translations.

### Real translations on an Apple silicon Mac

For a smaller local model, install the translation libraries into the same
environment and start NLLB-600M:

```
cd mt-game
.venv/bin/python -m pip install 'torch>=2.14.1' 'transformers==4.57.6' 'sentencepiece>=0.2.2'
HF_HOME="$PWD/.venv/huggingface" .venv/bin/python server.py \
  --model nllb-600m --via es,fr,de --explain confidence,backtranslation
```

The first launch downloads about 2.5 GB of model weights into the local,
gitignored environment. CUDA is selected when available, otherwise Apple
GPU acceleration (MPS), otherwise CPU. MPS uses half-precision weights to
reduce memory use. Add `--device cpu` to explicitly use the CPU. After the
model is cached, `HF_HUB_OFFLINE=1` can be added to the launch command to
run without contacting the model hub.

Open http://localhost:8000/dev. NLLB provides real translations, confidence,
alternatives, history, and back-translation, but does not support `--explain cot`.
See the Chinese vocabulary limitation below when choosing a model for a study.

## Settings

`play.sh` runs `server.py` with the default setup. Extra flags go at the end,
e.g. `GPU=1 ./play.sh --explain cot,backtranslation`.

| flag | meaning |
|---|---|
| `--model` | translator: `seed-x` (default in play.sh, can explain itself), `translategemma`, `nllb-600m/1.3b/3.3b`, or `mock` (no model, for testing the UI) |
| `--device` | `cuda:0`, `mps` (Apple GPU), or `cpu`; by default select CUDA, then MPS, then CPU according to availability |
| `--explain` | which explanations to show, comma-separated (default in play.sh: `cot`). See below. |
| `--via` | ordered intermediate languages from director to matcher, e.g. `es,fr,de`; replies reverse the order. Default: empty (direct translation). Editable per room in the developer view. |
| `--error-rate` | fraction of messages (0–1) that get a deliberate error before translating: a spatial word is swapped (left/right, top/bottom, big/small…). For testing whether people notice wrong translations. Default 0. |
| `--set` | default tangram set: `classic` or `abstract` |

Explanations (`--explain`):

| name | who sees it | what it shows |
|---|---|---|
| `cot` | receiver | the translator's reasoning about its translation (Seed-X only) |
| `history` | receiver | links to earlier messages that reused the same words |
| `confidence` | receiver | underlines words the model was unsure of |
| `alternatives` | receiver | other translations the model considered |
| `backtranslation` | sender | the translation translated back into the sender's language |

Notes on the models:
- NLLB's vocabulary is missing common Chinese characters (兔 rabbit, 脖/颈
  neck), which it silently drops. Avoid it for Chinese.
- TranslateGemma has no reasoning mode.
- Seed-X's reasoning is written after the translation, so it is a
  justification, not a trace of how the translation was made.

## Tangram sets

From [KiloGram](https://github.com/lil-lab/kilogram) (Ji et al. 2022), listed
in `tangram_set.json`:

- **classic:** the 12 figures used in most reference-game studies, all
  human-like.
- **abstract:** 12 shapes that people rarely agree on a name for.

The images are licensed for educational and research use only, so they are
not in this repo; `tangrams.py` downloads them into `tangrams/`.

## Logs

Each game writes `logs/<room>.jsonl`: joins, every message (original, MT
input, translation, back-translation, word probabilities), each explanation
as it finishes, every placement, each round's score and time, and any
deliberate errors. Messages also include `via` (configured intermediate
languages), `route` (actual ordered language codes), and `hops` (each step's
input, output in `text`, source/target, word probabilities, alternatives,
and latency in seconds). `relay_changed` records changes to the room's
route; `translation_failed` records the stage and step of a failed message.
The default room is `tangram-figure`.

## Tests

From `mt-game/`, with Flask installed:

```
.venv/bin/python -m unittest discover -s tests -v
```

Tests use the mock translator and in-memory tangram fixtures. They require
no GPU, model download, or tangram download, and write logs to a temporary
directory. They cover forward/reverse chains, direct translation, repeated
languages, intermediate outputs, explanation scope, route persistence,
failure/retry behavior, and changes attempted during translation.

## Files

| file | what it does |
|---|---|
| `server.py` | game server: rooms, rounds, scoring, logging, the message pipeline |
| `translate.py` | translator backends (Seed-X, TranslateGemma, NLLB, mock) |
| `relay.py` | language-route validation and sequential translation with per-step results |
| `explain.py` | receiver-side explanation modes |
| `inject.py` | deliberate errors (`--error-rate`) |
| `tangrams.py`, `tangram_set.json` | tangram sets and download |
| `static/index.html` | the player page |
| `static/dev.html` | the developer view |
| `play.sh` | starts the default setup |
| `tests/` | relay and game API regression tests without model dependencies |
