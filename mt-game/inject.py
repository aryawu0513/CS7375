"""Adversarial error injection (off unless --inject > 0).

Swaps a spatial or orientation word in the sender's text before translation,
so the receiver gets a fluent, confident, wrong translation. The sender's
back-translation (if on) will show the swapped word, which lets you test
whether senders catch it.
"""
import random, re

SWAPS = {
    "en": [("left", "right"), ("top", "bottom"), ("up", "down"), ("above", "below"),
           ("upper", "lower"), ("front", "back"), ("first", "last"), ("big", "small"),
           ("wide", "narrow"), ("tall", "short")],
    "zh": [("左", "右"), ("上", "下"), ("前", "后"), ("大", "小"), ("宽", "窄"), ("高", "矮")],
}


def perturb(text, lang, rng):
    """Returns (new_text, (old, new)) or (text, None) if nothing to swap."""
    pairs = SWAPS.get(lang, [])
    swap = {a: b for a, b in pairs} | {b: a for a, b in pairs}
    if lang == "zh":
        spots = [(m.start(), m.group()) for m in re.finditer("|".join(swap), text)]
    else:
        spots = [(m.start(), m.group()) for m in re.finditer(r"\b(%s)\b" % "|".join(swap), text, re.I)]
    if not spots:
        return text, None
    i, w = rng.choice(spots)
    new = swap[w.lower()]
    if w[:1].isupper():
        new = new.capitalize()
    return text[:i] + new + text[i + len(w):], (w, new)


def maybe_perturb(text, lang, p, rng=random):
    if p <= 0 or rng.random() >= p:
        return text, None
    return perturb(text, lang, rng)
