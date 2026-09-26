"""What backbones fit this MIL structure on a 10 GB card, and how fast.

One study is MAX_SERIES * SLICES = 64 images, so a "batch of 1" is already 64
forward passes. Activation memory, not parameter count, is what decides whether
a backbone is usable here -- which is why resnet34 was the starting point and
why swapping in something modern is not free.

DINO-pretrained weights are the interesting candidates: self-supervised
pretraining on ~1.7B images transfers better to medical imaging than ImageNet-1k
classification, and timm ships ConvNeXt variants with DINOv3 weights, which
avoids the data hunger of a plain ViT on 4407 noisy-labelled studies.

    python scripts/bench_backbones.py --size 384

Run it when the GPU is otherwise idle: it allocates most of the card.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.config import MAX_SERIES, SLICES  # noqa: E402
from rsna_knee.model import KneeStudyModel, masked_bce  # noqa: E402

# (timm name, note). ImageNet baselines first so the comparison has an anchor.
CANDIDATES = [
    ("resnet34", "current baseline, ImageNet-1k"),
    ("convnext_tiny.fb_in1k", "ConvNeXt-T, ImageNet-1k"),
    ("convnext_tiny.dinov3_lvd1689m", "ConvNeXt-T, DINOv3 on 1.69B images"),
    ("convnext_small.dinov3_lvd1689m", "ConvNeXt-S, DINOv3"),
    ("tf_efficientnetv2_s.in21k_ft_in1k", "EfficientNetV2-S"),
]

# 3525 training studies per fold at --folds 5.
STUDIES_PER_EPOCH = 3525


def trial(name: str, note: str, size: int, batch: int, ckpt: bool,
          steps: int = 2) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = torch.device("cuda")
    model = opt = vol = None
    try:
        model = KneeStudyModel(name, pretrained=False,
                               grad_checkpointing=ckpt).to(dev)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        vol = torch.randn(batch, MAX_SERIES, SLICES, size, size, device=dev)
        msk = torch.ones(batch, MAX_SERIES, dtype=torch.bool, device=dev)
        slots = torch.arange(MAX_SERIES, device=dev).expand(batch, MAX_SERIES)
        tgt = torch.rand(batch, 12, device=dev)
        t = time.time()
        for i in range(steps + 1):
            if i == 1:  # discard the warm-up step
                torch.cuda.synchronize()
                t = time.time()
            with torch.autocast("cuda", torch.float16):
                loss = masked_bce(model(vol, msk, slots), tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        per = (time.time() - t) / steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        hrs = STUDIES_PER_EPOCH / batch * per * 8 / 3600
        print(f"  {name:34s} {params:6.1f}M  peak {peak:5.2f} GB  "
              f"{per:5.2f} s/step  {hrs:4.1f} h per fold   {note}", flush=True)
    except torch.OutOfMemoryError:
        print(f"  {name:34s}  OOM at batch={batch} ckpt={int(ckpt)}", flush=True)
    except Exception as exc:
        print(f"  {name:34s}  {type(exc).__name__}: {str(exc)[:70]}", flush=True)
    finally:
        del model, opt, vol
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)}  {free/1e9:.1f}/{total/1e9:.1f} GB free")
    print(f"size={args.size} batch={args.batch} "
          f"({MAX_SERIES * SLICES * args.batch} images per step), "
          f"8 epochs per fold\n")
    for name, note in CANDIDATES:
        trial(name, note, args.size, args.batch, args.grad_checkpointing)


if __name__ == "__main__":
    main()
