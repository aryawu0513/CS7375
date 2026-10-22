"""Receiver-side explanations of a translated message.

The hard part of the tangram task is keeping shared names for figures
("the bird") stable across the MT. Yamashita & Ishida (CSCW '06) found MT
breaks this two ways: the same phrase is translated differently each time,
and a partner's reuse of your term comes back in different words. The modes
below give the receiver evidence about what the partner actually wrote.

Each mode: fn(m, room_msgs, mt) -> dict for the UI (or None), where m is the
message being sent (orig, mt_input, shown, mt, src, tgt, from;
mt_input differs from orig only when inject.py added an error) and room_msgs are earlier
messages.

  none         - baseline
  confidence   - highlight low-probability words in the translation
  alternatives - other beam translations the model considered
  history      - which earlier messages this one repeats or echoes, with a
                 stable gloss of each repeated phrase
  cot          - the model's own explanation of its translation (seed-x/mock)
"""
from translate import is_cjk, tokens

LOW_CONF = 0.5
MAX_LINKS = 3


def confidence(m, room_msgs, mt):
    return dict(words=m["mt"]["words"], threshold=LOW_CONF)


def alternatives(m, room_msgs, mt):
    alts = m["mt"]["alts"]
    return dict(alts=alts) if alts else None


def cot(m, room_msgs, mt):
    # Conditioned on the shown translation, but written after it, so it is a
    # rationale, not a faithful trace of how `shown` was produced.
    if not getattr(mt, "supports_cot", False):
        raise ValueError("cot needs --model seed-x or mock")
    # slow, so the server runs it after the message is delivered; partial text
    # is written into the message as it streams in
    def update(text, done=False):
        m["explain"]["cot"] = dict(text=text, done=done)

    update("")
    text = mt.explain(m["mt_input"], m["src"], m["tgt"], m["shown"], on_chunk=update)
    update(text, done=True)
    return m["explain"]["cot"]


def _ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _meaningful(g):
    # single Latin/Hangul words must be >= 4 letters ("bird", not "the");
    # CJK needs >= 2 characters
    if all(is_cjk(t) for t in g):
        return len(g) >= 2
    return len(g) >= 2 or len(g[0]) >= 4


def _shared(a, b, max_n=6):
    """Longest non-overlapping n-grams of token list a that also occur in b."""
    hits, used = [], set()
    for n in range(min(max_n, len(a)), 0, -1):
        bg = _ngrams(b, n)
        for i in range(len(a) - n + 1):
            g = tuple(a[i:i + n])
            span = set(range(i, i + n))
            if g in bg and not span & used and _meaningful(g):
                hits.append(g)
                used |= span
    return hits


def _join(g):
    return "".join(g) if all(is_cjk(t) for t in g) else " ".join(g)


def history(m, room_msgs, mt):
    """Two kinds of links, newest earlier message first:
    repeat - sender reuses words from their own earlier message
    echo   - sender reuses words from the receiver's message as the sender
             saw it in translation, so it likely refers to the receiver's term
    """
    cur = tokens(m["mt_input"])
    links, seen = [], set()
    for k in reversed(room_msgs):
        if k["from"] == m["from"]:
            kind, other = "repeat", tokens(k["mt_input"])
        else:
            kind, other = "echo", tokens(k["shown"])
        for g in _shared(cur, other):
            if g in seen:
                continue
            seen.add(g)
            phrase = _join(g)
            link = dict(kind=kind, msg_id=k["id"], round=k["round"])
            if kind == "repeat":
                # translating the phrase alone is deterministic, so the same
                # source term always gets the same gloss across messages
                link["gloss"] = mt.translate(phrase, m["src"], m["tgt"])["text"]
                link["earlier"] = k["shown"]
            else:
                link["yours"] = k["orig"]
            links.append(link)
            if len(links) >= MAX_LINKS:
                return dict(links=links)
    return dict(links=links) if links else None


SLOW = {"cot"}  # computed after delivery, in a background thread

MODES = dict(none=lambda *a: None, confidence=confidence, alternatives=alternatives,
             history=history, cot=cot)
