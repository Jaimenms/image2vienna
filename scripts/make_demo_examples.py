"""Build the demo examples the static Space offers: four drawn logos and six EUIPO marks.

uv run python scripts/make_demo_examples.py

Writes ``evals/demo_images/`` and ``evals/demo_examples.jsonl``.

The EUIPO marks come from ``evals/l3d_300.jsonl`` (L3D, CC BY 4.0): one per pictorial
category where the package put the office's division among its top 3 on the cached
description, so the demo shows the pipeline at its typical best. Descriptions under
the ``inventory`` prompt are taken from the cache or written by the default describer
(Ollama must be running for the drawn logos). Gold codes are kept in the JSONL for
the record but never reach the page.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from image2vienna.classifier import ViennaClassifier
from image2vienna.config import (
    DESCRIBER_PROMPTS,
    HEAVY_DESCRIBER,
    LIGHT_DESCRIBER,
    default_describer,
)
from image2vienna.describe import PROMPTS, get_describer
from image2vienna.eval import load_cases
from image2vienna.eval.describe import description_key
from image2vienna.scheme.codes import truncate_code

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "evals" / "demo_images"
TARGET = ROOT / "evals" / "demo_examples.jsonl"
W = 512

#: Category of the pictorial element to show, in gallery order.
PICK = [
    ("3", "Animals"),
    ("24", "Heraldry"),
    ("5", "Plants"),
    ("1", "Celestial bodies"),
    ("2", "Human beings"),
    ("7", "Buildings"),
]


def _star(d, cx, cy, r, fill):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=fill)


def draw_logos() -> list[dict]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = []
    im = Image.new("RGB", (W, W), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((120, 200, 400, 480), fill="navy")
    d.ellipse((190, 170, 450, 430), fill="white")
    for cx in (160, 256, 352):
        _star(d, cx, 110, 42, "goldenrod")
    im.save(IMAGES / "stars_moon.png")
    out.append(("stars_moon.png", "Three stars over a crescent moon", ["1.1.4", "1.7.6"]))

    im = Image.new("RGB", (W, W), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 300, W, W), fill="steelblue")
    for i in range(12):
        a = math.pi + i * math.pi / 11
        d.line((256, 300, 256 + 230 * math.cos(a), 300 + 230 * math.sin(a)), fill="orange", width=8)
    d.pieslice((156, 200, 356, 400), 180, 360, fill="orange")
    im.save(IMAGES / "sunrise_sea.png")
    out.append(("sunrise_sea.png", "Sun rising over the sea", ["1.3.1", "6.3.1"]))

    im = Image.new("RGB", (W, W), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 472, 472), outline="black", width=10)
    d.rounded_rectangle((200, 170, 312, 420), radius=30, fill="darkgreen")
    d.rectangle((236, 90, 276, 190), fill="darkgreen")
    d.rectangle((228, 80, 284, 100), fill="black")
    im.save(IMAGES / "bottle_circle.png")
    out.append(("bottle_circle.png", "A bottle inside a circle", ["19.7.1", "26.1.1"]))

    im = Image.new("RGB", (W, W), "white")
    d = ImageDraw.Draw(im)
    d.polygon([(120, 140), (392, 140), (392, 330), (256, 460), (120, 330)], fill="firebrick")
    d.polygon(
        [(150, 130), (150, 60), (205, 110), (256, 40), (307, 110), (362, 60), (362, 130)],
        fill="goldenrod",
    )
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 150)
    except OSError:
        font = ImageFont.load_default()
    d.text((256, 290), "AB", fill="white", font=font, anchor="mm")
    im.save(IMAGES / "shield_crown_ab.png")
    out.append(
        (
            "shield_crown_ab.png",
            "Letters AB on a shield under a crown",
            ["24.1.1", "24.9.9", "27.5.1"],
        )
    )
    return [
        {
            "id": f"demo:{name[:-4]}",
            "image": f"demo_images/{name}",
            "title": title,
            "vienna": codes,
            "source": "drawn with Pillow",
            "descriptions": {},
        }
        for name, title, codes in out
    ]


def pick_euipo(clf: ViennaClassifier) -> list[dict]:
    cases = load_cases(ROOT / "evals" / "l3d_300.jsonl")
    default_key = description_key(default_describer())
    inventory_key = description_key(default_describer(), "inventory")
    chosen = []
    used = set()
    for cat, label in PICK:
        for c in cases:
            if c.id in used or default_key not in c.descriptions:
                continue
            gold_div = {truncate_code(g, "division") for g in c.vienna if g.startswith(cat + ".")}
            if not gold_div:
                continue
            top = clf.classify_text(c.descriptions[default_key], level="division", top_k=3)
            if gold_div & {m.code for m in top}:
                used.add(c.id)
                src = c.image_path()
                shutil.copy2(src, IMAGES / src.name)
                chosen.append(
                    {
                        "id": c.id,
                        "image": f"demo_images/{src.name}",
                        "title": f"{label}: {c.text}" if c.text else label,
                        "vienna": list(c.vienna),
                        "source": "EUIPO open data via the Large Labelled Logo Dataset (CC BY 4.0)",
                        "descriptions": {
                            k: v
                            for k, v in c.descriptions.items()
                            if k in (default_key, inventory_key)
                        },
                    }
                )
                break
    return chosen


def main():
    clf = ViennaClassifier("10")
    examples = draw_logos() + pick_euipo(clf)
    cached = {}
    if TARGET.exists():  # keep descriptions already written
        for line in TARGET.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cached[d["id"]] = d.get("descriptions", {})
    for ex in examples:
        ex["descriptions"] = {**cached.get(ex["id"], {}), **ex["descriptions"]}
    # both models the demo names: heavy (Qwen2.5-VL via Ollama) and light (SmolVLM, the
    # model the page itself runs), each under the prompt it follows best
    for mode, spec in (("heavy", HEAVY_DESCRIBER), ("light", LIGHT_DESCRIBER)):
        prompt_name = DESCRIBER_PROMPTS[mode]
        key = description_key(spec, prompt_name)
        describer = None
        for ex in examples:
            if key in ex["descriptions"]:
                continue
            describer = describer or get_describer(spec)
            ex["descriptions"][key] = describer.describe(
                IMAGES / Path(ex["image"]).name, prompt=PROMPTS[prompt_name]
            )
            print("described", ex["id"], "with", spec, prompt_name)
    with open(TARGET, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(examples)} examples -> {TARGET}")


if __name__ == "__main__":
    main()
