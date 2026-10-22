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
- A session is 4 rounds with the same figures in a new order each round, so
  the pair can build up shared names ("the kneeling man"). Scores show after
  each round.
- Each player types in their own language. The receiver sees only the
  translation.

What each side sees besides the translation, in the default setup:

| who | what | why |
|---|---|---|
| sender | a **back-translation** of what the partner received | "did my meaning survive?" |
| receiver | the **translator's reasoning** (chain of thought) under each message | "why did it say this?" |

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
  MT model received (highlighted if an error was injected), what the
  receiver saw, the sender's back-translation, the translator's reasoning,
  and how long it took.
- **Controls:** language dropdowns for each side (then **Load**), the
  tangram set (`classic` or `abstract`; picking one starts a new game), and
  **New game**.

The developer view shows the answer, so never give it to participants.

## Settings

`play.sh` runs `server.py` with the default setup. Extra flags go at the end,
e.g. `GPU=1 ./play.sh --inject 0.2`.

| flag | meaning |
|---|---|
| `--model` | translator: `seed-x` (default in play.sh, can explain itself), `translategemma`, `nllb-600m/1.3b/3.3b`, or `mock` (no model, for testing the UI) |
| `--explain` | what the receiver sees, comma-separated: `cot` (reasoning), `history` (links to earlier messages that reused the same words), `confidence` (underlines words the model was unsure of), `alternatives` (other translations it considered), `none` |
| `--sender-bt` | show the sender a back-translation |
| `--inject P` | with probability P, swap a spatial word (left/right, top/bottom, big/small…) before translating: a deliberate error |
| `--set` | default tangram set: `classic` or `abstract` |
| `--rounds` | rounds per game (default 4) |

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
as it finishes, every placement, and each round's score and time. The
default room is `tangram-figure`.

## Files

| file | what it does |
|---|---|
| `server.py` | game server: rooms, rounds, scoring, logging, the message pipeline |
| `translate.py` | translator backends (Seed-X, TranslateGemma, NLLB, mock) |
| `explain.py` | receiver-side explanation modes |
| `inject.py` | deliberate error injection |
| `tangrams.py`, `tangram_set.json` | tangram sets and download |
| `static/index.html` | the player page |
| `static/dev.html` | the developer view |
| `play.sh` | starts the default setup |
