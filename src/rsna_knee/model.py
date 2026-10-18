"""Study-level model: slices -> series -> study -> 12 logits.

The label is attached to the study, but the evidence sits in a few slices of one
or two series: an ACL tear shows on sagittal slices through the notch, a Baker's
cyst on posterior axial slices. That is a multiple-instance problem at two levels,
so the model pools twice with attention rather than averaging:

    slices of a series  --attention-->  series embedding
    series of a study   --attention-->  study embedding  -->  12 logits

Attention is per-target. A single shared pooling would force one slice ranking to
serve all twelve findings, but the slices that matter for "Baker's" are not the
ones that matter for "PF OA", and the gate should differ accordingly.
"""

from __future__ import annotations

import timm
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F

from .config import MAX_SERIES, SIZE, SLICES
from .labels import TARGETS

N_TARGETS = len(TARGETS)


class TargetAttentionPool(nn.Module):
    """Attention pooling that learns a separate gate per target.

    Input  (B, N, D) features and a (B, N) validity mask.
    Output (B, T, D) -- one pooled embedding per target.
    """

    def __init__(self, dim: int, n_targets: int = N_TARGETS, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (B, N, T) attention logits, masked so padding cannot win attention.
        a = self.score(x)
        a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        # A row with no valid entries would give NaN after softmax; keep it at zero.
        empty = ~mask.any(dim=1)
        a = torch.softmax(a, dim=1)
        a = torch.nan_to_num(a, nan=0.0)
        pooled = torch.einsum("bnt,bnd->btd", a, x)
        return pooled.masked_fill(empty[:, None, None], 0.0)


class SeriesEncoder(nn.Module):
    """2.5D encoder: each slice is seen with its neighbours as RGB channels.

    Stacking adjacent slices into the three input channels lets an ordinary
    ImageNet backbone see through-plane continuity, which is what separates a
    meniscal tear from a partial-volume artefact, without the cost of 3D convs.
    """

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 drop: float = 0.0, grad_checkpointing: bool = False,
                 chunk: int = 0):
        super().__init__()
        # Images per backbone call. See forward() for why this is the lever that
        # decides whether a heavy backbone fits at all.
        self.chunk = int(chunk)
        extra = {}
        if backbone.startswith(("vit_", "eva", "deit")):
            # Plain ViTs fix their input size at build time (DINOv2 ships at
            # 518); building at the corpus size resamples the position
            # embeddings once instead of failing on every 384 px slice.
            extra["img_size"] = SIZE
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3,
            drop_rate=drop, **extra,
        )
        if grad_checkpointing:
            # The MIL structure means one study is MAX_SERIES * SLICES = 64
            # images, so activation memory, not parameter count, is what fills
            # the card. Recomputing activations trades ~30% speed for roughly
            # half the memory, which is what makes 384 fit a 10 GB card at all.
            try:
                self.net.set_grad_checkpointing(True)
            except Exception:
                pass
        self.dim = self.net.num_features

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """(B, S, H, W) uint8-scaled volume -> (B, S, D) per-slice features."""
        b, s, h, w = vol.shape
        # Neighbouring slices as channels, clamped at the stack ends.
        prev = torch.cat([vol[:, :1], vol[:, :-1]], dim=1)
        nxt = torch.cat([vol[:, 1:], vol[:, -1:]], dim=1)
        x = torch.stack([prev, vol, nxt], dim=2).reshape(b * s, 3, h, w)

        if not self.chunk or x.shape[0] <= self.chunk:
            return self.net(x).reshape(b, s, self.dim)

        # One study is 64 images, and they all go through the backbone before the
        # MIL pooling can run, so the whole study's activations are live at once.
        # That is what puts a heavy backbone over a 16 GB card even with the
        # backbone's own checkpointing on.
        #
        # Splitting the call is not enough by itself: autograd would still hold
        # every chunk's checkpoint boundaries simultaneously, so the stored total
        # would be unchanged. Each chunk is therefore wrapped in its own
        # checkpoint, which leaves only the chunk's input and its (B*S, D)
        # features alive between forward and backward -- features are ~200 KB for
        # a whole study. Peak then tracks one chunk instead of the study, and the
        # cost is one extra recompute of chunks that were going to be recomputed
        # anyway.
        if self.training and torch.is_grad_enabled():
            outs = [checkpoint(self.net, c, use_reentrant=False)
                    for c in x.split(self.chunk)]
        else:
            # Nothing is stored for backward under no_grad, so checkpointing
            # would only cost time. Chunking still caps the transient peak.
            outs = [self.net(c) for c in x.split(self.chunk)]
        return torch.cat(outs, dim=0).reshape(b, s, self.dim)


class KneeStudyModel(nn.Module):
    """Full study-level model over a fixed grid of (series slots x slices)."""

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 n_targets: int = N_TARGETS, drop: float = 0.1,
                 grad_checkpointing: bool = False, chunk: int = 0,
                 n_slots: int = MAX_SERIES):
        super().__init__()
        self.encoder = SeriesEncoder(backbone, pretrained, drop=drop,
                                     grad_checkpointing=grad_checkpointing,
                                     chunk=chunk)
        dim = self.encoder.dim

        self.slice_pool = TargetAttentionPool(dim, n_targets)
        # A series carries its own identity (which plane, fluid-sensitive or not);
        # the study-level gate should be able to prefer, say, sagittal PD when
        # ranking meniscal tears.
        # One embedding per slot. With compartment ROI there are more slots
        # than series, and each half gets its own identity so the per-target
        # gate can prefer, say, the lateral view when ranking Lateral OA.
        self.slot_embed = nn.Embedding(n_slots, dim)
        self.series_pool = TargetAttentionPool(dim, n_targets)

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        # One linear head per target, applied to that target's own pooled vector.
        self.head_w = nn.Parameter(torch.zeros(n_targets, dim))
        self.head_b = nn.Parameter(torch.zeros(n_targets))
        nn.init.trunc_normal_(self.head_w, std=0.02)

    def forward(self, vol: torch.Tensor, series_mask: torch.Tensor,
                slots: torch.Tensor) -> torch.Tensor:
        """
        vol         (B, M, S, H, W) float, M series slots per study
        series_mask (B, M) bool, False where a study lacks that slot
        slots       (B, M) long, which slot each position is
        returns     (B, T) logits
        """
        b, m, s, h, w = vol.shape
        feats = self.encoder(vol.reshape(b * m, s, h, w))          # (B*M, S, D)

        # Every slice of a present series is valid; absent series are masked out
        # entirely at the series level below.
        slice_mask = series_mask.reshape(b * m, 1).expand(b * m, s)
        per_series = self.slice_pool(feats, slice_mask)            # (B*M, T, D)

        dim = per_series.shape[-1]
        per_series = per_series.reshape(b, m, -1, dim)
        per_series = per_series + self.slot_embed(slots)[:, :, None, :]

        # Pool across series independently for each target.
        t = per_series.shape[2]
        x = per_series.permute(0, 2, 1, 3).reshape(b * t, m, dim)
        mask = series_mask[:, None, :].expand(b, t, m).reshape(b * t, m)
        pooled = self.series_pool(x, mask)                         # (B*T, T, D)
        # Each target reads its own row, giving (B, T, D).
        idx = torch.arange(t, device=vol.device).repeat(b)
        study = pooled[torch.arange(b * t, device=vol.device), idx].reshape(b, t, dim)

        study = self.drop(self.norm(study))
        return (study * self.head_w).sum(-1) + self.head_b


def masked_bce(logits: torch.Tensor, targets: torch.Tensor,
               weight: torch.Tensor | None = None,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE against soft targets, ignoring NaN entries.

    Weak labels are graded in [0, 1] rather than binary, and soft-target BCE is
    the right objective: it trains "large effusion" harder than "trace effusion"
    instead of collapsing both to 1.

    `pos_weight` upweights the positive term per target. The metric is *macro*
    AUC, so every target counts 1/12 regardless of how rare it is -- but twelve
    targets share one backbone, and at 4.7% positive MCL contributes almost
    nothing to an unweighted loss. It is the worst-scoring target at 0.611, and
    lifting it to 0.75 would be worth more macro AUC (+0.012) than any plausible
    gain on the targets already above 0.85.
    """
    valid = torch.isfinite(targets)
    if not valid.any():
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(targets), reduction="none",
        pos_weight=pos_weight,
    )
    if weight is not None:
        loss = loss * weight
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def prevalence_pos_weight(rates, cap: float = 8.0, device=None) -> torch.Tensor:
    """Inverse-prevalence weights, capped so a rare target cannot dominate.

    (1 - p) / p is the weight that equalises the positive and negative terms.
    Uncapped that is ~20x for MCL, which destabilises training and drags the
    common targets down; 8 recovers most of the benefit without that.
    """
    p = torch.as_tensor(list(rates), dtype=torch.float32, device=device)
    p = p.clamp(min=1e-3, max=1 - 1e-3)
    return ((1 - p) / p).clamp(max=cap)
