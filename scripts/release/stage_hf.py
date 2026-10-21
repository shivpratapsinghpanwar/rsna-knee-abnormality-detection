"""Stage the final checkpoints for a Hugging Face model repo, hardlinked so nothing is copied."""
import os
import shutil
from pathlib import Path

W = Path(r"D:\kaggle_c\work")
REPO = Path(r"D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection")
OUT = Path(r"D:\kaggle_c\hf_upload")

shutil.rmtree(OUT, ignore_errors=True)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


for fam, src_dir, stem, folds in (
        ("coatnet_rmlp_2_rw_384", W / "arm_weights", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", range(5)),
        ("vit_large_patch14_dinov2", W / "dinol_gcp", "vit_large_patch14_dinov2.lvd142m", (0, 1, 2, 4))):
    for k in folds:
        link(src_dir / f"fold{k}_{stem}.pt", OUT / fam / f"fold{k}_{stem}.pt")
        link(src_dir / f"fold{k}_history.json", OUT / fam / f"fold{k}_history.json")

for p in (REPO / "src" / "rsna_knee").glob("*.py"):
    link(p, OUT / "code" / "rsna_knee" / p.name)
for name in ("infer.py", "oof.py", "train.py"):
    link(REPO / "scripts" / name, OUT / "code" / "scripts" / name)
link(REPO / "requirements.txt", OUT / "code" / "requirements.txt")

for p in sorted((W / "oof2").glob("*_f[0-4].csv")):
    link(p, OUT / "oof" / p.name)
link(W / "oof2" / "fold_cloud.csv", OUT / "oof" / "fold_cloud.csv")

total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"staged {sum(1 for p in OUT.rglob('*') if p.is_file())} files, {total / 1e9:.2f} GB in {OUT}")
