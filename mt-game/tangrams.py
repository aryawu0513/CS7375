"""KiloGram tangrams (Ji et al., EMNLP 2022; github.com/lil-lab/kilogram).

The images are for educational and research use only, so they are not in
this repo. `python tangrams.py` downloads every set listed in tangram_set.json
into tangrams/ (gitignored). The default set is KiloGram's page-A..page-L.

Figures are shown as solid silhouettes: KiloGram draws each of the 7 pieces
light gray with a white outline, which would give away the piece structure.
"""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SVG_DIR = os.path.join(HERE, "tangrams")
URL = "https://raw.githubusercontent.com/lil-lab/kilogram/main/dataset/tangrams-svg/{}"


def tangram_sets(path=os.path.join(HERE, "tangram_set.json")):
    with open(path) as f:
        return json.load(f)


def tangram_ids(name=None):
    cfg = tangram_sets()
    return cfg["sets"][name or cfg["default"]]["ids"]


def silhouette(svg):
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r'\s(fill|stroke)="[^"]*"', r' \1="currentColor"', svg)
    svg = re.sub(r'\sid="[^"]*"', "", svg)  # 12 inline SVGs would repeat ids 1..7
    return svg.replace("strokewidth=", "stroke-width=")


def load_set(ids):
    figs = []
    for i in ids:
        path = os.path.join(SVG_DIR, f"{i}.svg")
        if not os.path.exists(path):
            sys.exit(f"missing {path}; run `uv run python tangrams.py` to download the tangrams")
        with open(path) as f:
            figs.append(silhouette(f.read()))
    return figs


def download(ids):
    os.makedirs(SVG_DIR, exist_ok=True)
    for name in [f"{i}.svg" for i in ids] + ["COPYRIGHT-COVER-SHEET.pdf"]:
        dst = os.path.join(SVG_DIR, name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(URL.format(name), dst)
    print(f"{len(ids)} tangrams in {SVG_DIR} (educational/research use only; see COPYRIGHT-COVER-SHEET.pdf)")


if __name__ == "__main__":
    download(sorted({i for st in tangram_sets()["sets"].values() for i in st["ids"]}))
