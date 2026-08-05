"""Generate the 'why-slow' candidate-construction diagrams.

  dflash_pipeline.png - DFlash: no tree. Diffusion drafter -> Top-K (argmax),
                        one GPU op, straight to output. 0.05 ms/round.
  ddtree_pipeline.png - DDTree: Diffusion drafter -> Top-K (sort, upfront) ->
                        copy one table to CPU -> iterative heap build loop ->
                        output. 0.49 ms/round.

Run:  python3 assets/why-slow/make_why_slow.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent

# palette (matches assets/make_figs.py)
BLUE = "#2a78d6"      # diffusion / GPU work
AQUA = "#1baf7a"      # CPU work
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

GPU_FILL = "#eaf2fc"   # light blue tint for the GPU container
GPU_EDGE = BLUE
CPU_FILL = "#e8f6f0"   # light aqua tint for the CPU container
CPU_EDGE = AQUA
NODE_FILL = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
})


def container(ax, x, y, w, h, label, fill, edge):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fill, edgecolor=edge, linewidth=2.2, linestyle=(0, (6, 3)),
    )
    ax.add_patch(box)
    ax.text(x + 0.14, y + h - 0.16, label, ha="left", va="top",
            fontsize=13, fontweight="bold", color=edge)


def node(ax, cx, cy, w, h, lines, sub=None, edge=INK, textcolor=INK):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=NODE_FILL, edgecolor=edge, linewidth=1.8,
    )
    ax.add_patch(box)
    if sub is None:
        ax.text(cx, cy, lines, ha="center", va="center",
                fontsize=12, fontweight="bold", color=textcolor)
    else:
        ax.text(cx, cy + 0.14, lines, ha="center", va="center",
                fontsize=12, fontweight="bold", color=textcolor)
        ax.text(cx, cy - 0.20, sub, ha="center", va="center",
                fontsize=9.5, color=INK2)


def arrow(ax, x0, y0, x1, y1, color=INK2, label=None, lw=2.0, labeldy=0.16):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
        shrinkA=2, shrinkB=2,
    ))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + labeldy, label,
                ha="center", va="bottom", fontsize=9.5, color=INK2)


def dflash():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # GPU container holding the drafter and Top-K
    container(ax, 0.4, 0.55, 7.4, 3.2, "GPU", GPU_FILL, GPU_EDGE)

    node(ax, 2.15, 2.0, 2.7, 1.25, "Diffusion\ndrafter", edge=BLUE, textcolor=INK)
    node(ax, 5.85, 2.0, 2.2, 1.25, "argmax", edge=BLUE, textcolor=INK)

    # drafter -> top-K (inside GPU)
    arrow(ax, 3.55, 2.0, 4.72, 2.0, color=BLUE)
    # top-K -> output (crosses GPU boundary)
    arrow(ax, 6.98, 2.0, 9.55, 2.0, color=INK2)

    node(ax, 10.6, 2.0, 2.0, 1.25, "Output", edge=INK, textcolor=INK)

    ax.set_title("DFlash: no tree", loc="left", x=0.045, y=0.94,
                 fontsize=14, fontweight="bold", color=INK)

    fig.tight_layout()
    fig.savefig(OUT / "dflash_pipeline.png", dpi=200)
    plt.close(fig)


def ddtree():
    fig, ax = plt.subplots(figsize=(11.6, 3.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # GPU container
    container(ax, 0.4, 0.7, 7.3, 3.4, "GPU", GPU_FILL, GPU_EDGE)
    node(ax, 2.15, 2.25, 2.7, 1.25, "Diffusion\ndrafter", edge=BLUE, textcolor=INK)
    node(ax, 5.75, 2.25, 2.3, 1.25, "Top-K", sub="sort, upfront", edge=BLUE, textcolor=INK)
    arrow(ax, 3.55, 2.25, 4.58, 2.25, color=BLUE)

    # copy one table GPU -> CPU
    arrow(ax, 6.92, 2.25, 8.85, 2.25, color=INK2, label="copy to CPU")

    # CPU container
    container(ax, 8.9, 0.7, 4.9, 3.4, "CPU", CPU_FILL, CPU_EDGE)
    node(ax, 11.35, 2.05, 2.7, 1.35, "Build heap", edge=AQUA, textcolor=INK)

    # self loop above the heap-build node: iterates
    loop = FancyArrowPatch(
        (10.75, 2.78), (11.95, 2.78),
        connectionstyle="arc3,rad=-1.1",
        arrowstyle="-|>", mutation_scale=15, color=AQUA, lw=2.0,
    )
    ax.add_patch(loop)
    ax.text(11.35, 3.62, "loop", ha="center", va="center",
            fontsize=9.5, fontstyle="italic", color=AQUA)
    ax.text(11.35, 1.02, "index into table (fast)", ha="center", va="center",
            fontsize=9.5, color=INK2)

    # CPU -> output
    arrow(ax, 12.7, 2.05, 14.35, 2.05, color=INK2)
    node(ax, 15.05, 2.05, 1.7, 1.25, "Output", edge=INK, textcolor=INK)

    ax.set_title("DDTree: tree built on CPU", loc="left", x=0.035, y=0.95,
                 fontsize=14, fontweight="bold", color=INK)

    fig.tight_layout()
    fig.savefig(OUT / "ddtree_pipeline.png", dpi=200)
    plt.close(fig)


def span_loop(ax, x0, x1, y, color, label, rad=0.32):
    """Loop arc spanning several nodes: arcs above and runs right->left."""
    loop = FancyArrowPatch(
        (x1, y), (x0, y),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=15, color=color, lw=2.0,
    )
    ax.add_patch(loop)
    # arc apex sits ~rad*span/2 above the line; drop the label just above it
    apex = y + abs(rad) * (x1 - x0) / 2
    ax.text((x0 + x1) / 2, apex + 0.14, label,
            ha="center", va="bottom", fontsize=9.5, fontstyle="italic",
            color=color)


def dspark():
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    # everything lives on the GPU
    container(ax, 0.4, 0.65, 10.1, 3.55, "GPU", GPU_FILL, GPU_EDGE)

    node(ax, 2.15, 2.05, 2.6, 1.25, "Diffusion\ndrafter", edge=BLUE, textcolor=INK)
    node(ax, 5.65, 2.05, 2.9, 1.25, "Markov head\n(bias)", edge=BLUE, textcolor=INK)
    node(ax, 8.95, 2.05, 2.1, 1.25, "argmax", edge=BLUE, textcolor=INK)

    arrow(ax, 3.45, 2.05, 4.15, 2.05, color=BLUE)   # drafter -> markov head
    arrow(ax, 7.15, 2.05, 7.85, 2.05, color=BLUE)   # markov head -> argmax

    # iterative loop over bias + argmax, one token at a time
    span_loop(ax, 5.65, 8.95, 2.68, BLUE,
              "loop: N small GPU ops (bias + argmax, per token)")

    # argmax (the sampler) -> output (crosses GPU boundary)
    arrow(ax, 10.05, 2.05, 11.85, 2.05, color=INK2)
    node(ax, 12.9, 2.05, 1.9, 1.25, "Output", edge=INK, textcolor=INK)

    ax.set_title("DSpark: no tree", loc="left", x=0.035, y=0.95,
                 fontsize=14, fontweight="bold", color=INK)

    fig.tight_layout()
    fig.savefig(OUT / "dspark_pipeline.png", dpi=200)
    plt.close(fig)


def sparklingtree():
    fig, ax = plt.subplots(figsize=(13.6, 3.6))
    ax.set_xlim(0, 17.6)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    # GPU only runs the diffusion drafter
    container(ax, 0.4, 0.65, 3.35, 3.55, "GPU", GPU_FILL, GPU_EDGE)
    node(ax, 2.05, 2.05, 2.5, 1.35, "Diffusion\ndrafter", edge=BLUE, textcolor=INK)

    # copy the full draft matrix GPU -> CPU
    arrow(ax, 3.35, 2.05, 4.95, 2.05, color=INK2, label="copy full matrix")

    # CPU does everything: heap + top-K + Markov, recomputed every pop
    container(ax, 5.0, 0.65, 10.3, 3.55, "CPU", CPU_FILL, CPU_EDGE)
    node(ax, 6.75, 2.05, 2.5, 1.35, "Build heap", edge=AQUA, textcolor=INK)
    node(ax, 9.85, 2.05, 2.7, 1.35, "Markov\nmodel", edge=AQUA, textcolor=INK)
    node(ax, 12.95, 2.05, 2.1, 1.35, "Top-K", edge=AQUA, textcolor=INK)

    arrow(ax, 8.0, 2.05, 8.50, 2.05, color=AQUA)     # heap -> markov
    arrow(ax, 11.2, 2.05, 11.90, 2.05, color=AQUA)   # markov -> top-K

    # loop over the whole CPU pipeline, recomputed on every pop
    span_loop(ax, 6.75, 12.95, 2.78, AQUA,
              "loop: every pop (Markov + Top-K recomputed on CPU)", rad=0.26)

    # CPU -> output
    arrow(ax, 14.0, 2.05, 15.85, 2.05, color=INK2)
    node(ax, 16.7, 2.05, 1.7, 1.25, "Output", edge=INK, textcolor=INK)

    ax.set_title("SparklingTree: full tree on CPU", loc="left", x=0.03, y=0.95,
                 fontsize=14, fontweight="bold", color=INK)

    fig.tight_layout()
    fig.savefig(OUT / "sparklingtree_pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    dflash()
    ddtree()
    dspark()
    sparklingtree()
    print("wrote:", OUT / "dflash_pipeline.png")
    print("wrote:", OUT / "ddtree_pipeline.png")
    print("wrote:", OUT / "dspark_pipeline.png")
    print("wrote:", OUT / "sparklingtree_pipeline.png")
