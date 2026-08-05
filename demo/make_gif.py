"""Render the 4-way streaming race GIF from results/demo.json.

2x2 terminal grid -- Autoregressive, DSpark, DDTree, SparklingTree -- all
streaming the same chat (alpaca-style) answer with real H100 timing, then a
final "leaderboard" card with the times. Bursty token commits ARE the real
per-round acceptance pattern, not an animation choice.

Run: python3 make_gif.py    -> results/race_4way.gif
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
DATA = json.loads((HERE / "results" / "demo.json").read_text())

PANES = [  # (key, label, accent) — fixed palette, color follows the method
    ("ar", "Autoregressive", "#149d8e"),
    ("dspark", "DSpark", "#bf7d15"),
    ("ddtree", "DDTree", "#4a7fd4"),
    ("st", "SparklingTree (ours)", "#e34948"),
]

W, H = 1280, 780
PAD = 14
HEADER_H = 88
GRID_TOP = HEADER_H + 6
PANE_W = (W - 3 * PAD) // 2
PANE_H = (H - GRID_TOP - 3 * PAD) // 2

BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#30363d"
DIM = "#8b949e"
TEXT = "#e6edf3"
GOOD = "#3fb950"

FONT = "/System/Library/Fonts/Menlo.ttc"
F_TITLE = ImageFont.truetype(FONT, 21)
F_SUB = ImageFont.truetype(FONT, 13)
F_BODY = ImageFont.truetype(FONT, 13)
F_STAT = ImageFont.truetype(FONT, 14)
F_CARD = ImageFont.truetype(FONT, 18)

CHAR_W = F_BODY.getlength("M")
WRAP = int((PANE_W - 26) // CHAR_W)
LINE_H = 18
MAX_LINES = (PANE_H - 62) // LINE_H

FPS = 20
HOLD_S = 0.8       # after last finisher, before the leaderboard
CARD_S = 3.0       # leaderboard hold


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


def draw_pane(d, x, y, label, accent, m, t):
    d.rounded_rectangle([x, y, x + PANE_W, y + PANE_H], radius=8,
                        fill=PANEL_BG, outline=BORDER, width=1)
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse([x + 12 + i * 16, y + 10, x + 21 + i * 16, y + 19], fill=c)
    d.text((x + 66, y + 8), label, font=F_STAT, fill=accent)

    done = t >= m["seconds"]
    elapsed = min(t, m["seconds"])
    shown = round(m["tokens"] * elapsed / m["seconds"]) if m["seconds"] else 0
    stat = f"{elapsed:5.2f}s  {shown:>3} tok"
    d.text((x + PANE_W - 12 - F_SUB.getlength(stat), y + 10), stat, font=F_SUB, fill=DIM)

    lines = wrap(text_at(m, t))
    if not done and int(t * FPS) % 8 < 5:
        lines = lines or [""]
        lines[-1] += "▍"
    yy = y + 36
    for ln in lines[-MAX_LINES:]:
        d.text((x + 13, yy), ln, font=F_BODY, fill=TEXT)
        yy += LINE_H
    if done:
        msg = f"✓ {m['seconds']:.2f}s · {m['tokens']} tok · {m['tps']:.0f} tok/s"
        d.text((x + 13, y + PANE_H - 26), msg, font=F_STAT, fill=GOOD)


def leaderboard(d, alpha_bg=True):
    if alpha_bg:
        d.rectangle([0, 0, W, H], fill=(13, 17, 23, 216))
    cw, ch = 640, 300
    cx, cy = (W - cw) // 2, (H - ch) // 2 + 20
    d.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=12,
                        fill=PANEL_BG, outline=BORDER, width=2)
    d.text((cx + 24, cy + 18), "Final times — same answer, same GPU", font=F_CARD, fill=TEXT)
    ranked = sorted(PANES, key=lambda p: DATA[p[0]]["seconds"])
    ar_tps = DATA["ar"]["tps"]
    max_tps = max(DATA[k]["tps"] for k, _, _ in PANES)
    y = cy + 64
    for key, label, accent in ranked:
        m = DATA[key]
        d.text((cx + 24, y), f"{label:<22}", font=F_STAT, fill=accent)
        bar_w = int(300 * m["tps"] / max_tps)
        d.rounded_rectangle([cx + 24, y + 22, cx + 24 + bar_w, y + 32], radius=4, fill=accent)
        stat = f"{m['seconds']:5.2f}s   {m['tps']:5.1f} tok/s   {m['tps']/ar_tps:4.2f}×"
        d.text((cx + cw - 24 - F_STAT.getlength(stat), y), stat, font=F_STAT, fill=TEXT)
        y += 56


def frame(t: float, show_card: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    d.text((PAD, 12), "One prompt, four speculators", font=F_TITLE, fill=TEXT)
    cfg = DATA["config"]
    d.text((PAD, 42), f"chat (alpaca-style) · {cfg['target']} · one H100 · "
                      f"tree budget {cfg['budget']} · greedy · identical output",
           font=F_SUB, fill=DIM)
    d.text((PAD, 62), "> " + DATA["prompt"][:150], font=F_SUB, fill=DIM)
    speed = DATA["st"]["tps"] / DATA["ar"]["tps"]
    tag = f"{speed:.1f}× vs AR"
    d.text((W - PAD - F_TITLE.getlength(tag), 12), tag, font=F_TITLE, fill="#e34948")

    pos = [(PAD, GRID_TOP), (PAD * 2 + PANE_W, GRID_TOP),
           (PAD, GRID_TOP + PANE_H + PAD), (PAD * 2 + PANE_W, GRID_TOP + PANE_H + PAD)]
    for (key, label, accent), (x, y) in zip(PANES, pos):
        draw_pane(d, x, y, label, accent, DATA[key], t)
    if show_card:
        leaderboard(d)
    return img


def main():
    race_end = max(DATA[k]["seconds"] for k, _, _ in PANES)
    n_race = int((race_end + HOLD_S) * FPS)
    frames = [frame(i / FPS, False) for i in range(n_race)]
    frames += [frame(race_end + HOLD_S, True)] * int(CARD_S * FPS)
    out = HERE / "results" / "race_4way.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
