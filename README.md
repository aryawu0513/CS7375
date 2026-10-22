# CS7375 project: explaining machine translation in collaboration

Class project on human-centered explainable AI. We study two people who don't
share a language and collaborate through a machine translator (MT) in the
middle, and ask what explanations of the translation actually help them.

## What's here

| folder | what it is |
|---|---|
| [`mt-game/`](mt-game/) | a two-player chat game, the **tangram figure task**, played through an MT model, with switchable explanations and a developer view. See its [README](mt-game/README.md) to run and play it. |

## Quick start

On the lab machine, with a free GPU:

```
cd mt-game
uv sync && uv run python tangrams.py   # once
GPU=1 ./play.sh
```

Then open the developer view to play both sides on one screen.