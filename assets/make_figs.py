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

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.25, 4.55)
    ax.axis("off")

    # --- the GPU bubble
    gpu = mpatches.FancyBboxPatch((0.25, 0.45), 10.1, 3.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.22",
                                  facecolor="#f4f3ef", edgecolor=BASELINE,
                                  linewidth=1.6, zorder=1)
    ax.add_patch(gpu)
    ax.text(0.55, 3.82, "GPU", color=INK, fontsize=13, fontweight="bold", zorder=3)

    def arrow(x1, x2, y, above=None, below=None):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                    mutation_scale=16), zorder=4)
        if above:
            ax.text((x1 + x2) / 2, y + 0.17, above, ha="center", va="bottom",
                    color=INK2, fontsize=8.5, family="monospace", zorder=4)
        if below:
            ax.text((x1 + x2) / 2, y - 0.17, below, ha="center", va="top",
                    color=INK2, fontsize=8.5, family="monospace", zorder=4)

    # --- stage 1: DFlash draft model
    dm = mpatches.FancyBboxPatch((0.55, 1.55), 1.42, 1.35,
                                 boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor=BLUE, edgecolor="none", zorder=3)
    ax.add_patch(dm)
    ax.text(1.26, 2.36, "DFlash", ha="center", color="#ffffff",
            fontweight="bold", fontsize=12.5, zorder=4)
    ax.text(1.26, 2.02, "draft model", ha="center", color="#dce9fb",
            fontsize=9.5, zorder=4)
    arrow(2.07, 2.62, 2.22)

    # --- stage 2: the full logits matrix, 15 x ~152k
    mx, my, mw, mh = 3.02, 1.28, 2.5, 1.9
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
    ax.text(mx + mw / 2, my - 0.24, r"vocab $\approx$ 152k cols", ha="center",
            color=INK2, fontsize=9, zorder=4)
    ax.text(mx - 0.13, my + mh / 2, "15 rows (positions 1 … 15)", ha="center",
            va="center", color=INK2, fontsize=8.5, rotation=90, zorder=4)
    arrow(mx + mw + 0.12, mx + mw + 0.92, 2.22,
          above="torch.topk(K)", below="logsumexp\n+ sort rows")

    # --- stage 3: sorted top-K slab (dark -> light, left to right, each row)
    sx, sy, sw, sh = 6.68, 1.28, 1.6, 1.9
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
    ax.text(sx + sw / 2, sy + sh + 0.30, "sorted top-K", ha="center",
            color=INK, fontsize=11, fontweight="bold", zorder=4)
    ax.text(sx + sw / 2, sy - 0.24, "K = tree token budget", ha="center",
            color=INK2, fontsize=9, zorder=4)
    ax.annotate("best", (sx + kw * 0.45, sy + sh + 0.07), ha="center",
                color=MUTED, fontsize=8, zorder=4)
    ax.annotate("worst", (sx + sw - kw * 0.45, sy + sh + 0.07), ha="center",
                color=MUTED, fontsize=8, zorder=4)

    # --- outputs
    arrow(sx + sw + 0.12, sx + sw + 0.52, 2.22)
    ox = sx + sw + 0.60
    for i, (name, desc) in enumerate([("top_logits", "values"),
                                      ("top_token_ids", "vocab indices")]):
        oy = 2.62 - i * 0.85
        ax.add_patch(mpatches.FancyBboxPatch((ox, oy - 0.28), 1.42, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     facecolor="#ffffff", edgecolor=BASELINE, lw=1.1, zorder=3))
        ax.text(ox + 0.71, oy + 0.10, name, ha="center", color=INK,
                fontsize=9, family="monospace", fontweight="bold", zorder=4)
        ax.text(ox + 0.71, oy - 0.14, desc, ha="center", color=MUTED,
                fontsize=8.5, zorder=4)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_sort_marginals.png", dpi=200)
    plt.close(fig)


def fig7():
    """Step 2: copy the sorted top-K tensors from GPU to CPU.

    Heap building + lookup tables are sequential -> CPU-friendly, but the
    copy is a device-to-host sync: the GPU stalls and pays transfer cost.
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.25, 4.15)
    ax.axis("off")

    def bubble(x, w, label):
        b = mpatches.FancyBboxPatch((x, 0.55), w, 3.15,
                                    boxstyle="round,pad=0.02,rounding_size=0.22",
                                    facecolor="#f4f3ef", edgecolor=BASELINE,
                                    linewidth=1.6, zorder=1)
        ax.add_patch(b)
        ax.text(x + 0.3, 3.32, label, color=INK, fontsize=13,
                fontweight="bold", zorder=3)

    def tensor_box(x, y, w, name, desc, fs=9.5):
        ax.add_patch(mpatches.FancyBboxPatch((x, y - 0.31), w, 0.68,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     facecolor="#ffffff", edgecolor=BASELINE, lw=1.1, zorder=3))
        ax.text(x + w / 2, y + 0.12, name, ha="center", color=INK,
                fontsize=fs, family="monospace", fontweight="bold", zorder=4)
        ax.text(x + w / 2, y - 0.14, desc, ha="center", color=MUTED,
                fontsize=8.5, zorder=4)

    bubble(0.3, 3.75, "GPU")
    bubble(6.55, 3.75, "CPU")

    y_hi, y_lo = 2.55, 1.45
    tensor_box(0.85, y_hi, 2.6, "top_logits", "values")
    tensor_box(0.85, y_lo, 2.6, "top_token_ids", "vocab indices")
    tensor_box(7.05, y_hi, 2.9, "top_log_probs_cpu", "float32, - log_z", fs=9)
    tensor_box(7.05, y_lo, 2.9, "top_token_ids_cpu", "long", fs=9)

    # sync barrier between the two devices
    ax.plot([5.38, 5.38], [0.75, 3.55], color=MUTED, lw=1.2,
            ls=(0, (4, 3)), zorder=2)
    ax.text(5.38, 3.72, "device -> host sync", ha="center", color=INK2,
            fontsize=9.5, fontweight="bold", zorder=4)

    def arrow(y, label):
        ax.annotate("", xy=(7.0, y), xytext=(3.52, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                    mutation_scale=16), zorder=4)
        ax.text(5.38, y + 0.15, label, ha="center", va="bottom", color=INK2,
                fontsize=8.5, family="monospace", zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    arrow(y_hi, '.to("cpu", float32)')
    arrow(y_lo, '.to("cpu", long)')

    fig.tight_layout()
    fig.savefig(OUT / "fig7_gpu_to_cpu_copy.png", dpi=200)
    plt.close(fig)


def fig8():
    """Step 3: best-first heap expansion over the sorted top-K matrix.

    Pop the best cumulative-probability candidate; push its next sibling
    (same depth, rank+1) and its best child (depth+1, rank 0 -- always the
    first column, never the cell straight below). Repeat until the node
    budget is spent. Runs on the CPU.
    """
    import matplotlib.patches as mpatches

    lo, hi = np.array([0xcd, 0xe2, 0xfb]), np.array([0x0d, 0x36, 0x6b])

    def ramp(t):
        rgb = lo + (hi - lo) * min(max(t, 0.0), 1.0)
        return "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))

    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.0, 5.45)
    ax.axis("off")

    # --- the CPU bubble around everything
    cpu = mpatches.FancyBboxPatch((0.28, 0.12), 10.04, 5.1,
                                  boxstyle="round,pad=0.02,rounding_size=0.22",
                                  facecolor="#f4f3ef", edgecolor=BASELINE,
                                  linewidth=1.6, zorder=0)
    ax.add_patch(cpu)
    ax.text(0.58, 4.87, "CPU", color=INK, fontsize=13, fontweight="bold",
            zorder=3)

    # --- the candidate heap (left)
    hx, hw = 0.6, 2.15
    ax.add_patch(mpatches.FancyBboxPatch((hx, 0.85), hw, 3.35,
                 boxstyle="round,pad=0.02,rounding_size=0.18",
                 facecolor="#ffffff", edgecolor=BASELINE, lw=1.5, zorder=1))
    ax.text(hx + hw / 2, 3.85, "candidate heap", ha="center", color=INK,
            fontsize=11, fontweight="bold", zorder=3)
    entries = [(3.15, 0.72, "best"), (2.55, 0.42, None),
               (1.95, 0.26, None), (1.35, 0.14, None)]
    for ey, t, tag in entries:
        ax.add_patch(mpatches.FancyBboxPatch((hx + 0.25, ey), hw - 0.5, 0.42,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=ramp(t), edgecolor="none", zorder=3))
        if tag:
            ax.text(hx + hw / 2, ey + 0.21, tag, ha="center", va="center",
                    color="#ffffff", fontsize=9, fontweight="bold", zorder=4)
    ax.text(hx + hw / 2, 1.28, "ordered by cumulative\nlog-prob of the path",
            ha="center", va="top", color=MUTED, fontsize=8, zorder=4)

    # --- sorted top-K matrix (right)
    cs, nrow, ncol = 0.56, 6, 7
    mx, mtop = 4.85, 4.05

    def cell_xy(r, c):
        return mx + c * cs, mtop - (r + 1) * cs

    in_tree = {(0, 0)}
    popped = (0, 1)
    sibling, child = (0, 2), (1, 0)

    for r in range(nrow):
        for c in range(ncol):
            x, y = cell_xy(r, c)
            if (r, c) in in_tree:
                fc = ramp(0.75)
            elif (r, c) == popped:
                fc = ramp(0.45)
            else:
                fc = ramp(0.05)
            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), cs - 0.06,
                                       cs - 0.06, facecolor=fc, zorder=3))
    for (r, c) in (sibling, child):
        x, y = cell_xy(r, c)
        ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), cs - 0.06, cs - 0.06,
                                   facecolor="none", edgecolor=BLUE, lw=1.8,
                                   ls=(0, (3, 2)), zorder=5))
    px, py = cell_xy(*popped)
    ax.add_patch(plt.Rectangle((px + 0.03, py + 0.03), cs - 0.06, cs - 0.06,
                               facecolor="none", edgecolor=INK, lw=2.0, zorder=5))

    ax.text(mx + ncol * cs + 0.12, mtop - 3 * cs, "…", color=MUTED,
            fontsize=13, zorder=4)

    # axes semantics
    ax.annotate("", xy=(mx + ncol * cs, 4.5), xytext=(mx, 4.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1,
                                mutation_scale=11), zorder=2)
    ax.text(mx + ncol * cs / 2, 4.62, r"rank (best $\rightarrow$ worst)",
            ha="center", color=INK2, fontsize=9.5, zorder=4)
    ax.annotate("", xy=(mx - 0.28, mtop - nrow * cs), xytext=(mx - 0.28, mtop),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1,
                                mutation_scale=11), zorder=4)
    ax.text(mx - 0.5, mtop - nrow * cs / 2, "depth (1 … 15)", ha="center",
            va="center", rotation=90, color=INK2, fontsize=9.5, zorder=4)

    # 2a. push sibling: popped -> (same depth, rank + 1)
    sxc, syc = cell_xy(*sibling)
    ax.annotate("", xy=(sxc + 0.08, syc + cs / 2),
                xytext=(px + cs - 0.04, py + cs / 2),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                mutation_scale=14), zorder=6)
    ax.annotate("a) push next best sibling (rank + 1)",
                xy=(sxc + cs / 2 + 0.14, syc + cs + 0.02), xytext=(7.35, 4.26),
                ha="left", va="center", color=INK2, fontsize=9, zorder=6,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=0.18"))

    # 2b. push child: popped -> (depth + 1, rank 0) -- down-LEFT to column 0
    cxc, cyc = cell_xy(*child)
    ax.annotate("", xy=(cxc + cs - 0.12, cyc + cs - 0.14),
                xytext=(px + 0.15, py + 0.13),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                mutation_scale=14), zorder=6)
    ax.annotate("b) push best child\n(depth + 1, rank 0)",
                xy=(cxc + 0.02, cyc + cs / 2 - 0.05), xytext=(2.98, 2.35),
                ha="left", va="center", color=INK2, fontsize=9, zorder=6,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=-0.15"))

    # legend
    lx, ly = 6.0, 0.42
    for dx, fc, ec, ls, lab in [
            (0.0, ramp(0.75), "none", "-", "in tree"),
            (1.35, ramp(0.45), INK, "-", "just popped"),
            (2.85, ramp(0.05), BLUE, (0, (3, 2)), "pushed to heap")]:
        ax.add_patch(plt.Rectangle((lx + dx, ly - 0.09), 0.24, 0.24,
                                   facecolor=fc, edgecolor=ec, lw=1.6,
                                   ls=ls, zorder=4))
        ax.text(lx + dx + 0.34, ly + 0.03, lab, va="center", color=INK2,
                fontsize=8.5, zorder=4)

    ax.text(0.62, 0.42, "repeat until node_count == budget", color=MUTED,
            fontsize=9, family="monospace", va="center", zorder=4)

    fig.tight_layout()
    fig.savefig(OUT / "fig8_best_first_heap.png", dpi=200)
    plt.close(fig)


def fig9():
    """Step 4: flatten the tree into a boolean visibility (ancestor) mask.

    Each tree node becomes a row; row i copies its parent's row (all of the
    parent's ancestors) and sets the diagonal bit. Node 4 and its ancestor
    path are highlighted in the tree and as the outlined rows in the mask.
    """
    import matplotlib.patches as mpatches

    lo, hi = np.array([0xcd, 0xe2, 0xfb]), np.array([0x0d, 0x36, 0x6b])

    def ramp(t):
        rgb = lo + (hi - lo) * min(max(t, 0.0), 1.0)
        return "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))

    # example tree: parents[i] for i = 0..5
    parents = [None, 0, 0, 1, 1, 2]
    n = len(parents)
    vis = np.zeros((n, n), dtype=bool)
    vis[0, 0] = True
    for i in range(1, n):
        vis[i, :i] = vis[parents[i], :i]
        vis[i, i] = True

    HILITE = 4          # the row whose construction we illustrate
    HPARENT = parents[HILITE]

    fig, ax = plt.subplots(figsize=(10.6, 5.3))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.0, 5.3)
    ax.axis("off")

    # --- the CPU bubble around everything
    cpu = mpatches.FancyBboxPatch((0.28, 0.12), 10.04, 4.95,
                                  boxstyle="round,pad=0.02,rounding_size=0.22",
                                  facecolor="#f4f3ef", edgecolor=BASELINE,
                                  linewidth=1.6, zorder=0)
    ax.add_patch(cpu)
    ax.text(0.58, 4.72, "CPU", color=INK, fontsize=13, fontweight="bold",
            zorder=3)

    # --- left: the accepted draft tree, nodes labeled by flat index
    pos = {0: (2.0, 4.05), 1: (1.2, 3.05), 2: (2.8, 3.05),
           3: (0.7, 2.05), 4: (1.7, 2.05), 5: (2.8, 2.05)}
    r = 0.26
    for i, p in enumerate(parents):
        if p is None:
            continue
        x1, y1 = pos[p]
        x2, y2 = pos[i]
        ax.plot([x1, x2], [y1 - r, y2 + r],
                color=INK if i == HILITE else BASELINE,
                lw=2.0 if i == HILITE else 1.4, zorder=1)
    for i, (x, y) in pos.items():
        hot = i == HILITE
        anc = vis[HILITE, i] and not hot
        fc = ramp(0.85) if hot else (ramp(0.45) if anc else "#ffffff")
        ax.add_patch(plt.Circle((x, y), r, facecolor=fc,
                                edgecolor=INK if hot else BASELINE,
                                lw=1.8 if hot else 1.3, zorder=3))
        ax.text(x, y, str(i), ha="center", va="center",
                color="#ffffff" if (hot or anc) else INK,
                fontweight="bold", fontsize=11, zorder=4)
    ax.text(2.0, 4.55, "draft tree (flat indices)", ha="center", color=INK,
            fontsize=11, fontweight="bold", zorder=4)

    # --- arrow to the mask
    ax.annotate("", xy=(4.65, 2.9), xytext=(3.65, 2.9),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                mutation_scale=16), zorder=4)
    ax.text(4.15, 3.08, "flatten", ha="center", color=INK2, fontsize=8.5,
            family="monospace", zorder=4)

    # --- right: the visibility matrix
    cs = 0.56
    mx, mtop = 5.6, 4.35
    for i in range(n):
        for j in range(n):
            x = mx + j * cs
            y = mtop - (i + 1) * cs
            fc = ramp(0.75) if vis[i, j] else ramp(0.04)
            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), cs - 0.06,
                                       cs - 0.06, facecolor=fc, zorder=3))
    # outline the highlighted row and its parent's row
    for row, lw_, ec, ls in [(HPARENT, 1.6, BLUE, (0, (3, 2))),
                             (HILITE, 2.0, INK, "-")]:
        y = mtop - (row + 1) * cs
        ax.add_patch(plt.Rectangle((mx + 0.015, y + 0.015),
                                   n * cs - 0.03, cs - 0.03,
                                   facecolor="none", edgecolor=ec, lw=lw_,
                                   ls=ls, zorder=5))

    # index labels
    for j in range(n):
        ax.text(mx + j * cs + cs / 2, mtop + 0.14, str(j), ha="center",
                color=MUTED, fontsize=9, zorder=4)
    for i in range(n):
        ax.text(mx - 0.16, mtop - (i + 1) * cs + cs / 2, str(i), ha="right",
                va="center", color=MUTED, fontsize=9, zorder=4)
    ax.text(mx + n * cs / 2, mtop + 0.46, "visibility mask  (row = query token,"
            " col = KV it may attend to)", ha="center", color=INK,
            fontsize=10.5, fontweight="bold", zorder=4)

    # legend
    lx, ly = 0.62, 0.42
    for dx, fc, lab in [(0.0, ramp(0.75), "True (ancestor)"),
                        (1.85, ramp(0.04), "False (masked)")]:
        ax.add_patch(plt.Rectangle((lx + dx, ly - 0.09), 0.24, 0.24,
                                   facecolor=fc, zorder=4))
        ax.text(lx + dx + 0.34, ly + 0.03, lab, va="center", color=INK2,
                fontsize=8.5, zorder=4)

    fig.tight_layout()
    fig.savefig(OUT / "fig9_tree_mask.png", dpi=200)
    plt.close(fig)


def fig10():
    """DSpark Markov-head correction: the generic loop step for positions 2-n.

    Position i's independent DFlash draft (fig4 column 2 numbers) enters the
    Markov head along with the previous token u_{i-1} = "No"; the head
    reweights it ("problem" +0.43, "course" -0.41) and the chosen u_i loops
    back as the conditioning token for position i+1. Position 1 has no
    previous token and is deliberately left out of this figure.
    """
    import matplotlib.patches as mpatches

    vocab = ["No", "problem", "of", "course"]
    before = [0.02, 0.45, 0.03, 0.50]
    after = [0.01, 0.88, 0.02, 0.09]
    GOOD = "#006300"   # delta-up text token
    BAD = "#d03b3b"    # delta-down

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.5, 5.15)
    ax.axis("off")
    SCALE = 2.1  # bar length at p = 1.0
    ys = [3.4, 2.7, 2.0, 1.3]

    # --- left: position i, independent DFlash draft
    ax.text(0.3, 4.6, r"position $i$ — DFlash draft", color=INK,
            fontsize=11.5, fontweight="bold")
    ax.text(0.3, 4.27, r"independent, for every $i \in 2 \ldots n$",
            color=INK2, fontsize=9)
    for wd, p, y in zip(vocab, before, ys):
        ax.text(1.55, y, f'"{wd}"', ha="right", va="center", color=INK, fontsize=10.5)
        ax.add_patch(plt.Rectangle((1.7, y - 0.16), p * SCALE, 0.32,
                                   facecolor=BLUE, alpha=0.35, zorder=3))
        ax.text(1.78 + p * SCALE, y, f"{p:.2f}", ha="left", va="center",
                color=INK2, fontsize=9)

    # --- middle: the Markov head + previous-token input
    bx, by, bw, bh = 4.75, 1.85, 2.35, 1.35
    ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
                 boxstyle="round,pad=0.02,rounding_size=0.14",
                 facecolor="#eceae4", edgecolor=BASELINE, lw=1.4, zorder=3))
    ax.text(bx + bw / 2, by + bh - 0.38, "Markov head", ha="center",
            color=INK, fontsize=11.5, fontweight="bold", zorder=4)
    ax.text(bx + bw / 2, by + 0.42, r"$\tilde{q}(u_i \mid u_{i-1})$",
            ha="center", color=INK2, fontsize=11, zorder=4)
    ax.text(bx + bw / 2, 4.6, r'previous token   $u_{i-1}$ = "No"',
            ha="center", color=INK, fontsize=10, zorder=4)
    ax.annotate("", xy=(bx + bw / 2, by + bh + 0.10),
                xytext=(bx + bw / 2, 4.38),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                mutation_scale=15), zorder=4)
    ax.annotate("", xy=(bx - 0.08, 2.52), xytext=(3.6, 2.52),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                mutation_scale=15), zorder=4)

    # --- right: position i reweighted, with deltas
    ax.text(8.1, 4.6, r"position $i$ — reweighted", color=INK,
            fontsize=11.5, fontweight="bold")
    ax.text(8.1, 4.27, r"conditioned on $u_{i-1}$", color=INK2, fontsize=9)
    x0 = 9.35
    for wd, pb, pa, y in zip(vocab, before, after, ys):
        ax.text(x0 - 0.15, y, f'"{wd}"', ha="right", va="center",
                color=INK, fontsize=10.5)
        ax.add_patch(plt.Rectangle((x0, y + 0.02), pb * SCALE, 0.15,
                                   facecolor=BLUE, alpha=0.22, zorder=3))
        ax.add_patch(plt.Rectangle((x0, y - 0.19), pa * SCALE, 0.21,
                                   facecolor=BLUE, zorder=3))
        vx = x0 + max(pb, pa) * SCALE + 0.08
        ax.text(vx, y - 0.04, f"{pa:.2f}", ha="left", va="center",
                color=INK2, fontsize=9)
        d = pa - pb
        ax.text(vx + 0.52, y - 0.04, f"({d:+.2f})", ha="left", va="center",
                color=GOOD if d > 0 else BAD, fontsize=9, fontweight="bold")
    ax.annotate("", xy=(8.95, 2.52), xytext=(bx + bw + 0.08, 2.52),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                mutation_scale=15), zorder=4)

    # legend: ghost = draft, solid = reweighted
    ax.add_patch(plt.Rectangle((9.35, 0.60), 0.3, 0.13, facecolor=BLUE,
                               alpha=0.22, zorder=3))
    ax.text(9.73, 0.66, "DFlash draft", color=INK2, fontsize=8.5, va="center")
    ax.add_patch(plt.Rectangle((9.35, 0.28), 0.3, 0.16, facecolor=BLUE, zorder=3))
    ax.text(9.73, 0.36, "after Markov head", color=INK2, fontsize=8.5,
            va="center")

    # --- the loop: chosen u_i becomes u_{i-1} for position i+1
    ax.annotate("", xy=(bx + bw / 2 + 0.55, by - 0.10), xytext=(9.05, 1.02),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5,
                                ls=(0, (4, 3)), mutation_scale=14,
                                connectionstyle="arc3,rad=-0.3"), zorder=4)
    ax.text(7.35, 0.42, "chosen $u_i$ becomes $u_{i-1}$\nfor position $i+1$",
            ha="center", color=MUTED, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig10_markov_conditioning.png", dpi=200)
    plt.close(fig)


def fig11():
    """First-order Markov model as a V x V bigram table.

    Rows = the given token u_{i-1}; each row is a distribution over the next
    token u_i (sums to 1). A visible 5-word corner of the table, with fading
    cells + arrows showing it continues to V ~ 152k in both directions, and
    the O(V^2) parameter-count punchline. Probabilities are illustrative;
    "No -> problem" and "of -> course" echo the fig4/fig10 example.
    """
    vocab = ["No", "problem", "of", "course", "the"]
    P = np.array([
        [0.05, 0.82, 0.04, 0.03, 0.06],   # given "No"
        [0.05, 0.02, 0.28, 0.05, 0.60],   # given "problem"
        [0.02, 0.03, 0.02, 0.85, 0.08],   # given "of"
        [0.20, 0.10, 0.15, 0.20, 0.35],   # given "course"
        [0.05, 0.35, 0.10, 0.25, 0.25],   # given "the"
    ])

    lo, hi = np.array([0xcd, 0xe2, 0xfb]), np.array([0x0d, 0x36, 0x6b])

    def cell_color(p):
        t = min(p / 0.9, 1.0)
        rgb = lo + (hi - lo) * t
        return "#{:02x}{:02x}{:02x}".format(*rgb.astype(int))

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.set_xlim(-2.9, 8.6)
    ax.set_ylim(9.8, -2.5)   # inverted: row 0 on top
    ax.axis("off")
    ax.set_aspect("equal")

    # axis headers
    ax.text(2.5, -2.05, r"next token  $u_i$", ha="center", color=INK,
            fontsize=12, fontweight="bold")
    ax.text(-2.55, 2.5, r"given  $u_{i-1}$", va="center", color=INK,
            fontsize=12, fontweight="bold", rotation=90)
    for c, wd in enumerate(vocab):
        ax.text(c + 0.5, -0.35, f'"{wd}"', ha="left", va="bottom", color=INK2,
                fontsize=9.5, rotation=35)
    for r, wd in enumerate(vocab):
        ax.text(-0.25, r + 0.5, f'"{wd}"', ha="right", va="center",
                color=INK2, fontsize=9.5)

    # the visible 5x5 corner
    for r in range(5):
        for c in range(5):
            p = P[r, c]
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.03), 0.94, 0.94,
                                       facecolor=cell_color(p), zorder=3))
            white = p > 0.35
            ax.text(c + 0.5, r + 0.5, f"{p:.2f}", ha="center", va="center",
                    color="#ffffff" if white else INK2,
                    fontweight="bold" if white else "normal",
                    fontsize=9.5, zorder=4)

    # fading continuation cells: right and down
    rng = np.random.default_rng(3)
    for r in range(5):
        for k, a in enumerate((0.45, 0.22, 0.09)):
            ax.add_patch(plt.Rectangle((5 + k + 0.03, r + 0.03), 0.94, 0.94,
                         facecolor=cell_color(rng.uniform(0.02, 0.4)),
                         alpha=a, zorder=3))
    for c in range(5):
        for k, a in enumerate((0.45, 0.22, 0.09)):
            ax.add_patch(plt.Rectangle((c + 0.03, 5 + k + 0.03), 0.94, 0.94,
                         facecolor=cell_color(rng.uniform(0.02, 0.4)),
                         alpha=a, zorder=3))
    ax.text(6.05, 6.05, r"$\ddots$", ha="center", va="center", color=MUTED,
            fontsize=15, zorder=4)

    # arrows: it keeps going to V ~ 152k
    ax.annotate("", xy=(8.35, 2.5), xytext=(5.4, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                mutation_scale=15), zorder=5)
    ax.text(6.9, 2.06, r"$V \approx 152\mathrm{k}$ cols", ha="center",
            color=INK2, fontsize=9.5, zorder=5)
    ax.annotate("", xy=(2.5, 8.35), xytext=(2.5, 5.4),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                mutation_scale=15), zorder=5)
    ax.text(2.75, 7.0, r"$V \approx 152\mathrm{k}$ rows", va="center",
            color=INK2, fontsize=9.5, zorder=5)

    fig.tight_layout()
    fig.savefig(OUT / "fig11_bigram_matrix.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    print("wrote figures to", OUT)
