"""Tangram-figure task: MT-mediated referential task (Clark & Wilkes-Gibbs style).

Director sees the target order; matcher must reproduce it. All chat goes
through the translator. Everything is logged to logs/<room>.jsonl.

  python server.py --model mock --explain cot,backtranslation
  open http://localhost:8000/?room=tangram-figure&role=director&lang=en
  open http://localhost:8000/?room=tangram-figure&role=matcher&lang=zh
  open http://localhost:8000/dev   (both sides on one page; shows the target)
"""
import argparse, json, os, random, threading, time

from flask import Flask, abort, jsonify, make_response, request, send_from_directory

from explain import MODES, SLOW
from inject import maybe_perturb
from tangrams import load_set, tangram_ids, tangram_sets
from translate import LANGS, MODELS, make_translator

p = argparse.ArgumentParser()
p.add_argument("--model", default="mock", choices=["mock", *MODELS])
p.add_argument("--device", default=None, help="e.g. cuda:0; default cuda if available")
p.add_argument("--explain", default="none",
               help="explanations, comma-separated. For the receiver: "
                    f"{','.join(x for x in MODES if x != 'none')}. For the sender: backtranslation")
p.add_argument("--alts", type=int, default=2, help="beam alternatives for --explain alternatives")
p.add_argument("--error-rate", type=float, default=0.0,
               help="fraction of messages (0-1) given a deliberate error before translation: "
                    "a spatial word is swapped (left/right, top/bottom...), to test whether people notice")
p.add_argument("--set", default=tangram_sets()["default"], choices=list(tangram_sets()["sets"]),
               help="tangram set from tangram_set.json")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--logdir", default="logs", help="where room logs go (use a scratch dir for tests)")
args = p.parse_args()
modes = [x for x in args.explain.split(",") if x != "none"]
SENDER_MODES = {"backtranslation"}
for x in modes:
    if x not in MODES and x not in SENDER_MODES:
        p.error(f"unknown explain mode {x}")
if "cot" in modes and args.model not in ("mock", "seed-x"):
    p.error("--explain cot needs --model seed-x (or mock)")

SETS = {name: load_set(tangram_ids(name)) for name in tangram_sets()["sets"]}
app = Flask(__name__, static_folder="static")
mt = make_translator(args.model, args.device)
rooms, lock = {}, threading.Lock()
os.makedirs(args.logdir, exist_ok=True)


def log(room_id, event, **kw):
    rec = dict(t=time.time(), room=room_id, event=event, **kw)
    with open(os.path.join(args.logdir, f"{room_id}.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_room(room_id, set_name=None):
    with lock:
        if room_id not in rooms:
            seed = random.randrange(1 << 30)
            set_name = set_name or args.set
            n = len(SETS[set_name])
            rooms[room_id] = dict(
                seed=seed,
                set=set_name,
                figures=SETS[set_name],
                round=1,
                target=random.sample(range(n), n),
                selection=[],
                langs={},
                messages=[],  # dicts with text per language
                pending={},  # role -> message still being translated
                results=[],
                round_start=time.time(),
            )
            log(room_id, "room_created", seed=seed, **(vars(args) | {"set": set_name}))
        return rooms[room_id]


def existing(room_id):
    """Rooms are created only by joining, so a stale tab polling an old room
    does not bring it back (and write a new log)."""
    r = rooms.get(room_id)
    if r is None:
        abort(make_response(jsonify(error="no such room; join first"), 404))
    return r


def other(role):
    return "matcher" if role == "director" else "director"


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/dev")
def dev():
    return send_from_directory("static", "dev.html")


@app.get("/api/meta")
def meta():
    """Choices for the dev view's dropdowns."""
    return jsonify(sets=list(SETS),
                   langs={code: name for code, (_, name) in LANGS.items()})


@app.get("/api/debug")
def debug():
    """Everything about a room, both sides. Shows the target order: dev only."""
    r = existing(request.args["room"])
    msgs = [{k: v for k, v in m.items() if k != "mt"} | {"alts": m["mt"]["alts"]}
            for m in r["messages"]]
    return jsonify(round=r["round"], langs=r["langs"], target=r["target"],
                   selection=r["selection"], results=r["results"], messages=msgs,
                   sets=list(SETS),
                   config=dict(model=args.model, set=r["set"], explain=modes,
                               error_rate=args.error_rate))


@app.post("/api/join")
def join():
    d = request.json
    r = get_room(d["room"])
    if d["lang"] not in LANGS:
        return jsonify(error=f"unknown lang {d['lang']}"), 400
    r["langs"][d["role"]] = d["lang"]
    log(d["room"], "join", role=d["role"], lang=d["lang"])
    return jsonify(ok=True)


@app.get("/api/state")
def state():
    room_id, role = request.args["room"], request.args["role"]
    r = existing(room_id)
    msgs = []
    for m in r["messages"]:
        mine = m["from"] == role
        item = dict(id=m["id"], mine=mine, round=m["round"],
                    text=m["orig"] if mine else m["shown"])
        if mine and m["back"] is not None:
            item["back"] = m["back"]
        if not mine:
            item["explain"] = m["explain"]
        msgs.append(item)
    # what is in flight, so both sides can see what they are waiting for
    pending = [dict(mine=p["from"] == role, stage=p["stage"],
                    text=p["orig"] if p["from"] == role else None)
               for p in list(r["pending"].values())]
    return jsonify(
        role=role,
        round=r["round"],
        figures=r["figures"],
        target=r["target"] if role == "director" else None,
        selection=r["selection"],
        partner_joined=other(role) in r["langs"],
        messages=msgs,
        pending=pending,
        game=r["seed"],  # changes on reset, so clients redraw the board
        results=r["results"],
        explain=modes,
    )


@app.post("/api/msg")
def msg():
    d = request.json
    r = existing(d["room"])
    role, text = d["role"], d["text"].strip()
    src = r["langs"].get(role, "en")
    tgt = r["langs"].get(other(role), src)
    t0 = time.time()
    r["pending"][role] = dict(orig=text, stage="translating", **{"from": role})
    try:
        mt_input, injected = maybe_perturb(text, src, args.error_rate)
        res = mt.translate(mt_input, src, tgt, num_alts=args.alts if "alternatives" in modes else 0)
        r["pending"][role]["stage"] = "back-translating"
        back = mt.translate(res["text"], tgt, src)["text"] if "backtranslation" in modes else None
    finally:
        r["pending"].pop(role, None)
    m = dict(id=len(r["messages"]), round=r["round"], **{"from": role},
             orig=text, mt_input=mt_input, injected=injected, shown=res["text"],
             back=back, src=src, tgt=tgt, mt=res)
    prev = list(r["messages"])
    m["explain"] = {x: MODES[x](m, prev, mt) for x in modes if x not in SLOW | SENDER_MODES}
    m["latency"] = round(time.time() - t0, 2)
    r["messages"].append(m)
    log(d["room"], "message", **m)
    slow = [x for x in modes if x in SLOW]
    if slow:
        threading.Thread(target=slow_explain, args=(d["room"], m, prev, slow, t0), daemon=True).start()
    return jsonify(ok=True)


def slow_explain(room_id, m, prev, slow, t0):
    for x in slow:
        m["explain"][x] = MODES[x](m, prev, mt)
        log(room_id, "explanation", msg_id=m["id"], mode=x, latency=round(time.time() - t0, 2),
            **{x: m["explain"][x]})


@app.post("/api/reset")
def reset():
    """Dev only: start the room over (new target order), keeping who joined.
    Optional "set" switches the tangram set."""
    room_id, set_name = request.json["room"], request.json.get("set")
    if set_name is not None and set_name not in SETS:
        return jsonify(error=f"unknown set {set_name}"), 400
    with lock:
        old = rooms.pop(room_id, None)
    r = get_room(room_id, set_name or (old or {}).get("set"))
    if old:
        r["langs"] = old["langs"]
    log(room_id, "reset", set=r["set"])
    return jsonify(ok=True)


@app.post("/api/select")
def select():
    d = request.json
    r = existing(d["room"])
    r["selection"] = [int(i) for i in d["selection"]][:len(r["figures"])]
    log(d["room"], "select", round=r["round"], selection=r["selection"])
    return jsonify(ok=True)


@app.post("/api/submit")
def submit():
    d = request.json
    if d["role"] != "matcher":
        return jsonify(error="only the matcher submits"), 403
    r = existing(d["room"])
    sel, tgt = r["selection"], r["target"]
    correct = sum(a == b for a, b in zip(sel, tgt))
    round_msgs = [m for m in r["messages"] if m["round"] == r["round"]]
    res = dict(round=r["round"], correct=correct, total=len(tgt),
               seconds=round(time.time() - r["round_start"], 1),
               n_messages=len(round_msgs),
               n_words=sum(len(m["orig"].split()) for m in round_msgs))
    r["results"].append(res)
    log(d["room"], "round_end", selection=sel, target=tgt, **res)
    # no round limit: same figures, new order, for as long as the pair plays
    r["round"] += 1
    r["target"] = random.sample(range(len(r["figures"])), len(r["figures"]))
    r["selection"] = []
    r["round_start"] = time.time()
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=True)
