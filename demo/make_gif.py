"""Render the 4-way streaming race GIF from results/demo.json.

Four VERTICAL columns ordered slowest -> fastest (left -> right), all streaming
the same long chat answer with real H100 timing. Finale: each column's text
blurs out and its TOTAL TIME + tok/s take over the pane. Bursty token commits
are the real per-round acceptance pattern.

Run: python3 make_gif.py    -> results/race_4way.gif
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
DATA = json.loads((HERE / "results" / "demo.json").read_text())

LABEL = {"ar": "Autoregressive", "dspark": "DSpark", "ddtree": "DDTree",
         "st": "SparklingTree"}
ACCENT = {"ar": "#149d8e", "dspark": "#bf7d15", "ddtree": "#4a7fd4", "st": "#e34948"}
# Column order: slowest -> fastest, from the measured trace itself.
ORDER = sorted(LABEL, key=lambda k: -DATA[k]["seconds"])

W, H = 1280, 760
PAD = 12
HEADER_H = 86
GRID_TOP = HEADER_H + 4
PANE_W = (W - 5 * PAD) // 4
PANE_H = H - GRID_TOP - PAD

BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#30363d"
DIM = "#8b949e"
TEXT = "#e6edf3"
GOOD = "#3fb950"

FONT = "/System/Library/Fonts/Menlo.ttc"
F_TITLE = ImageFont.truetype(FONT, 21)
F_SUB = ImageFont.truetype(FONT, 13)
F_BODY = ImageFont.truetype(FONT, 12)
F_STAT = ImageFont.truetype(FONT, 13)
F_BIG = ImageFont.truetype(FONT, 34)
F_MED = ImageFont.truetype(FONT, 18)

CHAR_W = F_BODY.getlength("M")
WRAP = int((PANE_W - 22) // CHAR_W)
LINE_H = 16
MAX_LINES = (PANE_H - 60) // LINE_H

FPS = 12
HOLD_S = 0.6
FINALE_S = 3.5


def wrap(text: str) -> list[str]:
    lines = []
    for para in text.split("\n"):
        while len(para) > WRAP:
            cut = para.rfind(" ", 0, WRAP)
            cut = cut if cut > WRAP // 2 else WRAP
            lines.append(para[:cut])
            para = para[cut:].lstrip()
        lines.append(para)
    return lines


def text_at(m: dict, t: float) -> str:
    cur = ""
    for ts, txt in m["checkpoints"]:
        if ts <= t:
            cur = txt
        else:
            break
    return cur


def pane_x(i: int) -> int:
    return PAD + i * (PANE_W + PAD)


def draw_pane(d, x, key, t):
    m = DATA[key]
    d.rounded_rectangle([x, GRID_TOP, x + PANE_W, GRID_TOP + PANE_H], radius=8,
                        fill=PANEL_BG, outline=BORDER, width=1)
    d.text((x + 12, GRID_TOP + 9), LABEL[key], font=F_STAT, fill=ACCENT[key])
    done = t >= m["seconds"]
    elapsed = min(t, m["seconds"])
    shown = round(m["tokens"] * elapsed / m["seconds"]) if m["seconds"] else 0
    stat = f"{elapsed:5.2f}s  {shown:>3}t"
    d.text((x + PANE_W - 10 - F_SUB.getlength(stat), GRID_TOP + 9), stat,
           font=F_SUB, fill=DIM)

    lines = wrap(text_at(m, t))
    if not done and int(t * FPS) % 6 < 4:
        lines = lines or [""]
        lines[-1] += "▍"
    yy = GRID_TOP + 34
    for ln in lines[-MAX_LINES:]:
        d.text((x + 11, yy), ln, font=F_BODY, fill=TEXT)
        yy += LINE_H
    if done:
        d.text((x + 11, GRID_TOP + PANE_H - 24), f"✓ {m['seconds']:.2f}s",
               font=F_STAT, fill=GOOD)


def header(d):
    d.text((PAD, 12), "One prompt, four speculators — slowest → fastest",
           font=F_TITLE, fill=TEXT)
    cfg = DATA["config"]
    d.text((PAD, 42), f"chat · {cfg['target']} · one H100 · tree budget {cfg['budget']} "
                      f"· greedy decoding · 450 tokens each", font=F_SUB, fill=DIM)
    d.text((PAD, 62), "> " + DATA["prompt"][:150], font=F_SUB, fill=DIM)
    speed = DATA["st"]["tps"] / DATA["ar"]["tps"]
    tag = f"{speed:.1f}× vs AR"
    d.text((W - PAD - F_TITLE.getlength(tag), 12), tag, font=F_TITLE, fill="#e34948")


def race_frame(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    header(d)
    for i, key in enumerate(ORDER):
        draw_pane(d, pane_x(i), key, t)
    return img


def finale_frame(t_end: float) -> Image.Image:
    img = race_frame(t_end)
    ar_tps = DATA["ar"]["tps"]
    for i, key in enumerate(ORDER):
        x = pane_x(i)
        # blur the pane body (text region), keep its header row sharp
        box = (x + 2, GRID_TOP + 30, x + PANE_W - 2, GRID_TOP + PANE_H - 2)
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(7)), box)
    d = ImageDraw.Draw(img, "RGBA")
    for i, key in enumerate(ORDER):
        m = DATA[key]
        x = pane_x(i)
        d.rectangle([x + 2, GRID_TOP + 30, x + PANE_W - 2, GRID_TOP + PANE_H - 2],
                    fill=(13, 17, 23, 140))
        cx = x + PANE_W // 2
        cy = GRID_TOP + PANE_H // 2 - 40
        big = f"{m['seconds']:.2f}s"
        d.text((cx - F_BIG.getlength(big) / 2, cy), big, font=F_BIG, fill=TEXT)
        med = f"{m['tps']:.0f} tok/s"
        d.text((cx - F_MED.getlength(med) / 2, cy + 48), med, font=F_MED,
               fill=ACCENT[key])
        rel = "baseline" if key == "ar" else f"{m['tps'] / ar_tps:.2f}× vs AR"
        d.text((cx - F_STAT.getlength(rel) / 2, cy + 78), rel, font=F_STAT, fill=DIM)
        if key == "st":
            d.rounded_rectangle([x, GRID_TOP, x + PANE_W, GRID_TOP + PANE_H],
                                radius=8, outline=ACCENT["st"], width=3)
            w = "fastest"
            d.text((cx - F_MED.getlength(w) / 2, cy + 106), w, font=F_MED,
                   fill=ACCENT["st"])
    return img


def main():
    race_end = max(DATA[k]["seconds"] for k in ORDER)
    n_race = int((race_end + HOLD_S) * FPS)
    frames = [race_frame(i / FPS) for i in range(n_race)]
    frames += [finale_frame(race_end + HOLD_S)] * int(FINALE_S * FPS)
    out = HERE / "results" / "race_4way.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
