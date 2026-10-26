"""Sequential translation through an ordered list of intermediate languages."""
import time

from translate import LANGS

MAX_VIA = 8


def validate_via(via):
    if not isinstance(via, list) or any(not isinstance(code, str) for code in via):
        raise ValueError("via must be a list of language codes")
    if len(via) > MAX_VIA:
        raise ValueError(f"choose at most {MAX_VIA} intermediate languages")
    for code in via:
        if code not in LANGS:
            raise ValueError(f"unknown intermediate language: {code}")
    return list(via)


def translation_route(src, tgt, via, reverse=False):
    """Reverse the relay for replies; collapse consecutive equal languages.

    Non-adjacent revisits are intentional (e.g. en -> fr -> en -> zh).
    A direct same-language message retains one identity step.
    """
    via = validate_via(via)
    if src not in LANGS or tgt not in LANGS:
        raise ValueError("unknown source or target language")
    route = [src]
    for code in [*(reversed(via) if reverse else via), tgt]:
        if code != route[-1]:
            route.append(code)
    return route if len(route) > 1 else [src, tgt]


def translate_chain(mt, text, route, num_alts=0, on_step=None):
    """Feed each output into the next step, retaining every result for audit.

    Alternatives and receiver confidence describe the last step only.
    Exceptions propagate: a failed relay must never deliver partial output.
    """
    if len(route) < 2 or any(code not in LANGS for code in route):
        raise ValueError("route must contain at least two supported language codes")
    hops = []
    total = len(route) - 1
    for step, (src, tgt) in enumerate(zip(route, route[1:]), 1):
        if on_step:
            on_step(step, total, src, tgt)
        start = time.monotonic()
        result = mt.translate(text, src, tgt, num_alts=num_alts if step == total else 0)
        hops.append(dict(step=step, src=src, tgt=tgt, input=text, **result,
                         latency=round(time.monotonic() - start, 3)))
        text = result["text"]
    return result, hops
