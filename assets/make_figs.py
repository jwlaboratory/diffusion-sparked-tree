"""Generate fig1/fig2/fig3 for the writeup.

  fig1_autoregressive_scaling.png  - AR decode time grows with sequence length
  fig2_diffusion_scaling.png       - diffusion decode time is ~flat in length
  fig3_acceptance_tradeoff.png     - Eagle-3 vs DFlash: acceptance rate vs tokens/step

Run:  python3 assets/make_figs.py

fig1/fig2 are schematic (no measured data). fig3 DFlash numbers come from
experiment1-harness/Results/summary.json (dflash.tree on GSM8K); Eagle-3 numbers
are the standard "short drafter" framing (drafts ~3, high per-token acceptance)
and can be swapped via the constants below.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent

# palette (validated defaults, light mode)
BLUE = "#2a78d6"      # diffusion / DFlash
AQUA = "#1baf7a"      # autoregressive / Eagle-3
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# fig3 inputs
DFLASH_ACCEPT_RATE = 0.87    # mean per-depth acceptance, dflash.tree, GSM8K (depths 1-15)
DFLASH_DRAFT_LEN = 15        # block size 16 -> 15 drafted after the anchor token
DFLASH_ACCEPTED = 8.7        # mean_accept, dflash.tree, GSM8K
EAGLE_ACCEPT_RATE = 0.95     # illustrative
EAGLE_DRAFT_LEN = 3
EAGLE_ACCEPTED = 2.6         # illustrative (0.95 chain over 3 drafts)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "font.size": 11,
})


def style_ax(ax):
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_xlim(0, 520)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 12, 2))
    ax.set_xlabel("Sequence length (tokens generated)")
    ax.set_ylabel("Wall-clock time (s)")


def fig1():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    n = np.linspace(0, 500, 200)
    # one forward pass per token; slight superlinear bend from growing KV/attention
    t = n * 0.017 + (n / 500) ** 2 * 1.6
    ax.plot(n, t, color=AQUA, lw=2.5, solid_capstyle="round")
    style_ax(ax)
    ax.annotate("Autoregressive", xy=(430, np.interp(430, n, t)),
                xytext=(300, 9.2), color=INK, fontweight="bold", fontsize=12)
    ax.annotate("1 sequential forward pass\nper token", xy=(300, np.interp(300, n, t)),
                xytext=(45, 6.4), color=INK2, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1,
                                shrinkA=8, shrinkB=6))
    ax.set_title("Autoregressive decoding: time grows with length",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_autoregressive_scaling.png", dpi=200)
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    n = np.linspace(0, 500, 200)
    t_ar = n * 0.017 + (n / 500) ** 2 * 1.6
    t_diff = 1.1 + n * 0.0012   # near-flat: fixed number of denoising passes
    ax.plot(n, t_ar, color=BASELINE, lw=1.6, ls=(0, (4, 3)))
    ax.plot(n, t_diff, color=BLUE, lw=2.5, solid_capstyle="round")
    style_ax(ax)
    ax.annotate("Diffusion", xy=(430, np.interp(430, n, t_diff)),
                xytext=(310, 2.4), color=INK, fontweight="bold", fontsize=12)
    ax.annotate("Autoregressive (ref.)", xy=(420, np.interp(420, n, t_ar)),
                xytext=(255, 9.4), color=INK2, fontsize=10)
    ax.annotate("all tokens denoised in parallel —\nfixed number of passes", xy=(250, 1.45),
                xytext=(120, 4.8), color=INK2, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_title("Diffusion decoding: time ~constant in length",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_diffusion_scaling.png", dpi=200)
    plt.close(fig)


def fig3():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 4.4))
    labels = ["Eagle-3", "DFlash"]
    colors = [AQUA, BLUE]
    x = np.array([0, 1])
    w = 0.52

    # left: per-token acceptance rate
    rates = [EAGLE_ACCEPT_RATE, DFLASH_ACCEPT_RATE]
    bars = axl.bar(x, rates, w, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, r in zip(x, rates):
        axl.text(xi, r + 0.02, f"{r:.0%}", ha="center", color=INK,
                 fontweight="bold", fontsize=12)
    axl.set_ylim(0, 1.08)
    axl.set_yticks(np.arange(0, 1.01, 0.25))
    axl.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    axl.set_title("Per-token acceptance rate", loc="left",
                  fontsize=12, fontweight="bold", color=INK, pad=10)

    # right: drafted (ghost) vs accepted (solid) tokens per step
    drafted = [EAGLE_DRAFT_LEN, DFLASH_DRAFT_LEN]
    accepted = [EAGLE_ACCEPTED, DFLASH_ACCEPTED]
    axr.bar(x, drafted, w, color=colors, alpha=0.22, zorder=2)
    axr.bar(x, accepted, w, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, d, a in zip(x, drafted, accepted):
        axr.text(xi, d + 0.45, f"{d} drafted", ha="center", color=INK2, fontsize=10)
        axr.text(xi, a - 0.45, f"{a:.1f} accepted", ha="center", va="top",
                 color="#ffffff", fontweight="bold", fontsize=11)
    axr.set_ylim(0, 17)
    axr.set_title("Tokens per drafting step", loc="left",
                  fontsize=12, fontweight="bold", color=INK, pad=10)

    for ax in (axl, axr):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, color=INK)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_color(BASELINE)

    fig.suptitle("Lower acceptance rate, but 5x the draft length -> more tokens per step",
                 x=0.065, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig3_acceptance_tradeoff.png", dpi=200)
    plt.close(fig)


def fig4():
    """DFlash block output: one independent distribution per position.

    Numbers are chosen so per-position argmax = ("No", "course") -> the
    incoherent "No course" pairing, to be annotated in a later figure.
    """
    vocab = ['"No"', '"problem"', '"of"', '"course"']
    probs = np.array([
        [0.55, 0.02],   # No
        [0.02, 0.45],   # problem
        [0.40, 0.03],   # of
        [0.03, 0.50],   # course
    ])

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    # sequential blue ramp, light (step 100) -> dark (step 700)
    lo, hi = np.array([0xcd, 0xe2, 0xfb]), np.array([0x0d, 0x36, 0x6b])

    def cell_color(p):
        t = p / 0.6  # scale so the top prob lands near the dark end
        rgb = lo + (hi - lo) * min(t, 1.0)
        return "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))

    for r in range(4):
        for c in range(2):
            p = probs[r, c]
            ax.add_patch(plt.Rectangle((c + 0.02, r + 0.02), 0.96, 0.96,
                                       facecolor=cell_color(p), zorder=2))
            ax.text(c + 0.5, r + 0.5, f"{p:.2f}", ha="center", va="center",
                    color="#ffffff" if p > 0.25 else INK2,
                    fontweight="bold" if p > 0.25 else "normal",
                    fontsize=13, zorder=3)

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([r"position 1  ($u_1$)", r"position 2  ($u_2$)"],
                       fontsize=11.5, color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([r + 0.5 for r in range(4)])
    ax.set_yticklabels(vocab, fontsize=12, color=INK)
    ax.tick_params(length=0)

    for c in range(2):
        ax.text(c + 0.5, 4.25, r"$\Sigma = 1.00$", ha="center",
                color=MUTED, fontsize=10)
    ax.set_ylim(4.5, 0)

    ax.set_title("DFlash output",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_dflash_output_table.png", dpi=200)
    plt.close(fig)


def fig5():
    """Chain vs tree at the same 4-token budget.

    Nodes show the cumulative (prefix) acceptance probability; edges carry the
    per-token probability q. Chain: "No problem at all", depth 4, E = 1.56.
    Tree: same "No"-prefix plus the "of course" branch, depth 2, E = 1.73.
    """
    lo, hi = np.array([0xcd, 0xe2, 0xfb]), np.array([0x0d, 0x36, 0x6b])

    def cell_color(p):
        t = min(p / 0.6, 1.0)
        rgb = lo + (hi - lo) * t
        return "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))

    def node(ax, x, y, word, cum, context=False):
        w, h = 1.7, 0.72
        fc = "#eceae4" if context else cell_color(cum)
        box = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                            edgecolor="none", zorder=3)
        box.set_path_effects([])
        ax.add_patch(box)
        if context:
            ax.text(x, y, word, ha="center", va="center", color=INK2,
                    fontsize=10.5, style="italic", zorder=4)
        else:
            white = cum > 0.25
            ink = "#ffffff" if white else INK
            ax.text(x, y + 0.11, f'"{word}"', ha="center", va="center",
                    color=ink, fontweight="bold", fontsize=11.5, zorder=4)
            ax.text(x, y - 0.17, f"cum {cum:.2f}", ha="center", va="center",
                    color=ink if white else INK2, fontsize = 9, zorder=4)

    def edge(ax, x1, y1, x2, y2, q):
        ax.plot([x1, x2], [y1 - 0.36, y2 + 0.36], color=BASELINE, lw=1.4, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 - 0.36 + y2 + 0.36) / 2
        ax.annotate(f"q = {q:.2f}", (mx, my), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center",
                    color=INK2, fontsize=9, zorder=4)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 5.6))
    for ax in (axl, axr):
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.9, 5.0)
        ax.axis("off")

    # --- chain: No -> problem -> at -> all, q = .55/.80/.70/.85
    axl.set_title("Chain (budget 4, depth 4)", loc="left",
                  fontsize=12.5, fontweight="bold", color=INK, pad=8)
    node(axl, 0, 4.5, "context  c", 0, context=True)
    chain = [("No", 0.55, 0.55), ("problem", 0.80, 0.44),
             ("at", 0.70, 0.31), ("all", 0.85, 0.26)]
    ys = [3.5, 2.5, 1.5, 0.5]
    edge(axl, 0, 4.5, 0, ys[0], chain[0][1])
    for k, (wd, q, cum) in enumerate(chain):
        node(axl, 0, ys[k], wd, cum)
        if k + 1 < len(chain):
            edge(axl, 0, ys[k], 0, ys[k + 1], chain[k + 1][1])
    axl.text(-2.5, -0.55, r"$\mathbb{E} = 0.55 + 0.44 + 0.31 + 0.26 = 1.56$",
             color=INK, fontsize=11.5)

    # --- tree: {No -> problem, of -> course}, same budget of 4 nodes
    axr.set_title("Tree (budget 4, depth 2)", loc="left",
                  fontsize=12.5, fontweight="bold", color=INK, pad=8)
    node(axr, 0, 4.5, "context  c", 0, context=True)
    edge(axr, 0, 4.5, -1.2, 3.5, 0.55)
    edge(axr, 0, 4.5, 1.2, 3.5, 0.40)
    node(axr, -1.2, 3.5, "No", 0.55)
    node(axr, 1.2, 3.5, "of", 0.40)
    edge(axr, -1.2, 3.5, -1.2, 2.5, 0.80)
    edge(axr, 1.2, 3.5, 1.2, 2.5, 0.85)
    node(axr, -1.2, 2.5, "problem", 0.44)
    node(axr, 1.2, 2.5, "course", 0.34)
    axr.text(-2.5, -0.55,
             r"$\mathbb{E} = 0.55 + 0.40 + 0.44 + 0.34 = \mathbf{1.73}$",
             color=INK, fontsize=11.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_chain_vs_tree.png", dpi=200)
    plt.close(fig)


def fig6():
    """Step 1 of tree construction: sorting the DFlash marginals, all on GPU.

    DFlash emits a 15 x ~152k logits matrix (one row per drafted position).
    torch.topk keeps the best K per row (K = tree token budget), logsumexp
    normalizes rows so logits are comparable, and each row ends up sorted.
    Output: top_logits (values) + top_token_ids (vocab indices).
    """
    import matplotlib.patches as mpatches

    lo, hi = np.array([0xcd, 0xe2, 0xfb]), np.array([0x0d, 0x36, 0x6b])

    def ramp(t):
        rgb = lo + (hi - lo) * min(max(t, 0.0), 1.0)
        return "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))

    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    # --- the GPU bubble
    gpu = mpatches.FancyBboxPatch((0.25, 0.45), 10.1, 3.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.22",
                                  facecolor="#f4f3ef", edgecolor=BASELINE,
                                  linewidth=1.6, zorder=1)
    ax.add_patch(gpu)
    ax.text(0.55, 3.82, "GPU", color=INK, fontsize=13, fontweight="bold", zorder=3)
    ax.text(1.32, 3.82, "(everything stays on-device)", color=MUTED,
            fontsize=9.5, zorder=3)

    def arrow(x1, x2, y, label=None, label2=None):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                    mutation_scale=16), zorder=4)
        if label:
            ax.text((x1 + x2) / 2, y + 0.18, label, ha="center", color=INK2,
                    fontsize=9, zorder=4)
        if label2:
            ax.text((x1 + x2) / 2, y - 0.22, label2, ha="center", color=MUTED,
                    fontsize=8.5, family="monospace", zorder=4)

    # --- stage 1: DFlash draft model
    dm = mpatches.FancyBboxPatch((0.62, 1.55), 1.5, 1.35,
                                 boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor=BLUE, edgecolor="none", zorder=3)
    ax.add_patch(dm)
    ax.text(1.37, 2.36, "DFlash", ha="center", color="#ffffff",
            fontweight="bold", fontsize=12.5, zorder=4)
    ax.text(1.37, 2.02, "draft model", ha="center", color="#dce9fb",
            fontsize=9.5, zorder=4)
    arrow(2.22, 2.92, 2.22, "one forward\npass")

    # --- stage 2: the full logits matrix, 15 x ~152k
    mx, my, mw, mh = 3.05, 1.28, 2.6, 1.9
    nrow = 8  # schematic rows standing in for 15
    rng = np.random.default_rng(7)
    ncol = 26
    cw, ch = mw / ncol, mh / nrow
    for r in range(nrow):
        for c in range(ncol):
            ax.add_patch(plt.Rectangle((mx + c * cw, my + r * ch),
                                       cw * 0.92, ch * 0.86,
                                       facecolor=ramp(rng.uniform(0.03, 0.5)),
                                       zorder=3))
    ax.add_patch(plt.Rectangle((mx - 0.045, my - 0.045), mw + 0.09, mh + 0.09,
                               facecolor="none", edgecolor=BASELINE,
                               linewidth=1.1, zorder=2))
    ax.text(mx + mw / 2, my + mh + 0.30, "draft logits", ha="center",
            color=INK, fontsize=11, fontweight="bold", zorder=4)
    # dimension braces
    ax.text(mx + mw / 2, my - 0.24, r"vocab $\approx$ 152k cols", ha="center",
            color=INK2, fontsize=9, zorder=4)
    ax.text(mx - 0.14, my + mh / 2, "15 rows\n(position\n1 … 15)", ha="right",
            va="center", color=INK2, fontsize=9, zorder=4)
    arrow(mx + mw + 0.12, mx + mw + 0.95, 2.22, "per row:",
          "topk(K) · logsumexp\n")
    ax.text(mx + mw + 0.535, 1.72, "sort", ha="center", color=MUTED,
            fontsize=8.5, family="monospace", zorder=4)

    # --- stage 3: sorted top-K slab (dark -> light, left to right, each row)
    sx, sy, sw, sh = 6.75, 1.28, 1.7, 1.9
    kcol = 9
    kw, kh = sw / kcol, sh / nrow
    for r in range(nrow):
        jitter = rng.uniform(-0.04, 0.04, size=kcol)
        for c in range(kcol):
            t = 0.62 * (1 - c / (kcol - 1)) + 0.05 + jitter[c]
            ax.add_patch(plt.Rectangle((sx + c * kw, sy + r * kh),
                                       kw * 0.9, kh * 0.86,
                                       facecolor=ramp(t), zorder=3))
    ax.add_patch(plt.Rectangle((sx - 0.045, sy - 0.045), sw + 0.09, sh + 0.09,
                               facecolor="none", edgecolor=BASELINE,
                               linewidth=1.1, zorder=2))
    ax.text(sx + sw / 2, sy + mh + 0.30, "sorted top-K", ha="center",
            color=INK, fontsize=11, fontweight="bold", zorder=4)
    ax.text(sx + sw / 2, sy - 0.24, "K = tree token budget", ha="center",
            color=INK2, fontsize=9, zorder=4)
    ax.annotate("best", (sx + kw * 0.45, sy + sh + 0.09), ha="center",
                color=MUTED, fontsize=8, zorder=4)
    ax.annotate("worst", (sx + sw - kw * 0.45, sy + sh + 0.09), ha="center",
                color=MUTED, fontsize=8, zorder=4)

    # --- outputs
    arrow(sx + sw + 0.12, sx + sw + 0.7, 2.22)
    ox = sx + sw + 0.78
    for i, (name, desc) in enumerate([("top_logits", "values"),
                                      ("top_token_ids", "vocab indices")]):
        oy = 2.62 - i * 0.85
        ax.add_patch(mpatches.FancyBboxPatch((ox, oy - 0.28), 1.62, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     facecolor="#ffffff", edgecolor=BASELINE, lw=1.1, zorder=3))
        ax.text(ox + 0.81, oy + 0.10, name, ha="center", color=INK,
                fontsize=9.5, family="monospace", fontweight="bold", zorder=4)
        ax.text(ox + 0.81, oy - 0.14, desc, ha="center", color=MUTED,
                fontsize=8.5, zorder=4)

    ax.set_title("Step 1 - sort the DFlash marginals (fast: small tensors, all on GPU)",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_sort_marginals.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("wrote fig1-fig6 to", OUT)
