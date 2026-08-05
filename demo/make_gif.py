"""Render the AR-vs-SparklingTree streaming GIF from results/demo.json.

DDTree-paper-style side-by-side terminal replay: both panes stream the same
gsm8k answer in real time; SparklingTree finishes ~5x sooner (and visibly
commits tokens in bursts -- that's speculative acceptance, not smoothing).

Run: python3 make_gif.py    -> results/ar_vs_sparklingtree.gif
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
DATA = json.loads((HERE / "results" / "demo.json").read_text())

# ---- layout ----------------------------------------------------------------- #
W, H = 1280, 640
PAD = 16
PANEL_W = (W - 3 * PAD) // 2
HEADER_H = 92
PANEL_TOP = HEADER_H + 8
PANEL_H = H - PANEL_TOP - PAD

BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#30363d"
DIM = "#8b949e"
TEXT = "#e6edf3"
AR_ACCENT = "#4a7fd4"
ST_ACCENT = "#e34948"
GOOD = "#3fb950"

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
F_TITLE = ImageFont.truetype(FONT_PATH, 21)
F_SUB = ImageFont.truetype(FONT_PATH, 13)
F_BODY = ImageFont.truetype(FONT_PATH, 14)
F_STAT = ImageFont.truetype(FONT_PATH, 15)

CHAR_W = F_BODY.getlength("M")
WRAP = int((PANEL_W - 28) // CHAR_W)
LINE_H = 20
MAX_LINES = (PANEL_H - 64) // LINE_H

FPS = 20
HOLD_S = 1.6


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


def text_at(method: dict, t: float) -> tuple[str, bool]:
    """(text_so_far, finished) at wall-time t."""
    ck = method["checkpoints"]
    cur = ""
    for ts, txt in ck:
        if ts <= t:
            cur = txt
        else:
            break
    return cur, t >= method["seconds"]


def draw_panel(d: ImageDraw.ImageDraw, x: int, name: str, accent: str,
               method: dict, t: float):
    d.rounded_rectangle([x, PANEL_TOP, x + PANEL_W, PANEL_TOP + PANEL_H],
                        radius=8, fill=PANEL_BG, outline=BORDER, width=1)
    # traffic dots + title
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse([x + 14 + i * 18, PANEL_TOP + 12, x + 24 + i * 18, PANEL_TOP + 22], fill=c)
    d.text((x + 74, PANEL_TOP + 10), name, font=F_STAT, fill=accent)

    body, done = text_at(method, t)
    shown_tokens = round(method["tokens"] * min(t, method["seconds"]) / method["seconds"]) \
        if method["seconds"] else 0
    elapsed = min(t, method["seconds"])
    tps = method["tps"]
    stat = f"{elapsed:5.2f}s   {shown_tokens:>3} tok   {tps:5.1f} tok/s"
    d.text((x + PANEL_W - 14 - F_SUB.getlength(stat), PANEL_TOP + 13), stat,
           font=F_SUB, fill=DIM)

    lines = wrap(body)
    cursor_visible = (not done) and (int(t * FPS) % 8 < 5)
    if cursor_visible:
        lines = lines or [""]
        lines[-1] += "▍"
    lines = lines[-MAX_LINES:]
    y = PANEL_TOP + 42
    for ln in lines:
        d.text((x + 14, y), ln, font=F_BODY, fill=TEXT)
        y += LINE_H
    if done:
        msg = f"✓ {method['tokens']} tokens in {method['seconds']:.2f}s — {tps:.0f} tok/s"
        d.text((x + 14, PANEL_TOP + PANEL_H - 30), msg, font=F_STAT, fill=GOOD)


def frame(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, 14), "Autoregressive vs SparklingTree", font=F_TITLE, fill=TEXT)
    cfg = DATA["config"]
    sub = (f"gsm8k · {cfg['target']} · one H100 · tree budget {cfg['budget']} "
           f"· greedy · identical output")
    d.text((PAD, 44), sub, font=F_SUB, fill=DIM)
    prompt = DATA["prompt"]
    prompt = prompt if len(prompt) < 150 else prompt[:147] + "..."
    d.text((PAD, 64), "> " + prompt[:160], font=F_SUB, fill=DIM)

    speed = DATA["st"]["tps"] / DATA["ar"]["tps"]
    tag = f"{speed:.1f}× faster"
    d.text((W - PAD - F_TITLE.getlength(tag), 14), tag, font=F_TITLE, fill=ST_ACCENT)

    draw_panel(d, PAD, "Autoregressive", AR_ACCENT, DATA["ar"], t)
    draw_panel(d, PAD * 2 + PANEL_W, "SparklingTree (ours)", ST_ACCENT, DATA["st"], t)
    return img


def main():
    total = max(DATA["ar"]["seconds"], DATA["st"]["seconds"]) + HOLD_S
    n = int(total * FPS)
    frames = [frame(i / FPS) for i in range(n)]
    out = HERE / "results" / "ar_vs_sparklingtree.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out} ({n} frames, {total:.1f}s, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
