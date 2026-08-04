"""Tree-aware round confidence: decide how much verification this round deserves.

The idea in one line: DSpark already ships a confidence head, but it predicts
CHAIN acceptance and `RESULTS.md` section 11 pointed it at per-depth width
allocation -- a coverage question it cannot answer. It was measured at -2.7%
acceptance / -6.6% speed and shelved. This is the same idea aimed at a question
it can actually answer, with the tree in the loop.

Two estimators live here, and they answer the same question at different cost:

`round_confidence` -- FREE. `base_draft_logits` is already materialized before
the tree is built, so this is one softmax and no extra forward pass. Its headline
field `pred_chain_len` is the drafter's own estimate of greedy-chain acceptance
under exactly the independence assumption DDTree already makes:

    E[accepted] = sum_{d=1..D} prod_{j<=d} p_j          p_j = top-1 prob at depth d

`TreeAcceptanceHead` -- TRAINED, and the part that "knows about the tree". The
free estimator above predicts what a CHAIN would accept. What the gate actually
wants is what the TREE will accept, and those differ most exactly where the
decision is hard. This head regresses tree acceptance from the same free
features, so it costs a 6-input MLP rather than a model pass.

Why the gate shrinks on CONFIDENT rounds, not uncertain ones
------------------------------------------------------------
`experiment4-speedup-verification/investigate/FINDINGS.md` finding 5 measured it:
rounds where the chain accepts <=3 tokens are 64% of rounds and carry 71% of
everything tree width produces. Cutting budget when the drafter is uncertain
would cut it exactly where the return lives. Saturated rounds are the free ones:
32% of DSpark's rounds, mean gain from width -0.32.

Nothing in this module influences tree contents unless a gate is explicitly
configured; `round_confidence` is pure measurement.
"""

import torch

# Field order of the tensor returned by round_confidence(). A module constant so
# the decode loop can stack rounds into one [rounds, len(FIELDS)] tensor and
# transfer ONCE at the end -- a per-round .item() would sync the device every
# round and distort the very timings this experiment measures.
FIELDS = (
    "pred_chain_len",   # expected greedy-chain acceptance; the predictor of record
    "root_p1",          # top-1 probability at depth 1
    "root_margin",      # top-1 minus top-2 log-prob at depth 1
    "root_entropy",     # entropy (nats) of the depth-1 distribution
    "mean_p1",          # mean top-1 probability across all depths
    "min_p1",           # weakest depth in the block
)

PRED_CHAIN_LEN = FIELDS.index("pred_chain_len")


def round_confidence(base_logits: torch.Tensor) -> torch.Tensor:
    """Confidence stats for one round. Returns [len(FIELDS)], no host sync.

    base_logits: [depth_limit, vocab] -- the drafter's per-depth next-token logits,
    i.e. exactly what gets handed to the tree builder.
    """
    if base_logits.ndim != 2:
        raise ValueError(f"expected [depth, vocab], got {tuple(base_logits.shape)}")
    if base_logits.shape[0] == 0:
        return torch.zeros(len(FIELDS), device=base_logits.device, dtype=torch.float32)

    logp = base_logits.float().log_softmax(dim=-1)          # [D, V]
    top2 = logp.topk(2, dim=-1).values                      # [D, 2]
    p1 = top2[:, 0].exp()                                   # [D]

    # prod_{j<=d} p_j in log space: cumulative sum of top-1 log-probs. Summing the
    # exponentials gives the expected run length before the first mismatch.
    pred_chain_len = torch.cumsum(top2[:, 0], dim=0).exp().sum()
    entropy = -(logp[0].exp() * logp[0]).sum()

    return torch.stack([
        pred_chain_len,
        p1[0],
        top2[0, 0] - top2[0, 1],
        entropy,
        p1.mean(),
        p1.min(),
    ])


def stack_rounds(rounds: list[torch.Tensor]) -> list[dict[str, float]]:
    """Collapse per-round tensors into plain dicts with ONE device transfer."""
    if not rounds:
        return []
    table = torch.stack(rounds).cpu().tolist()
    return [dict(zip(FIELDS, row)) for row in table]


class TreeAcceptanceHead(torch.nn.Module):
    """Regress THIS round's tree acceptance from the free confidence features.

    The tree-aware part. `pred_chain_len` estimates what a chain would accept;
    this estimates what the tree will accept, which is the quantity the gate
    needs. Six inputs, one hidden layer -- deliberately tiny, because
    FINDINGS.md finding 4 showed the gate only needs +-1.5-2 tokens of accuracy
    and anything larger would cost more than it saves.

    Trained offline on (features, accepted_length) pairs that a
    `measure_confidence=True` run already emits, so it needs no new data
    collection and no backbone gradients.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(len(FIELDS), hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        # Features are on wildly different scales (pred_chain_len spans 0-16,
        # probabilities 0-1, entropy 0-12). Standardize with buffers so the
        # normalization travels with the checkpoint instead of living in the
        # training script, where it would silently drift out of sync.
        self.register_buffer("feat_mean", torch.zeros(len(FIELDS)))
        self.register_buffer("feat_std", torch.ones(len(FIELDS)))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-6))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [..., len(FIELDS)] -> predicted tree acceptance, [...]."""
        x = (feats - self.feat_mean) / self.feat_std
        # Acceptance is >= 1 (the bonus token is always committed), so a softplus
        # floor keeps the head from wasting capacity learning that bound.
        return 1.0 + torch.nn.functional.softplus(self.net(x).squeeze(-1))
