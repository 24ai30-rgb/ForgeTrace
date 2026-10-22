"""Model-only evaluation of one or more U-Net checkpoints on every dataset in splits_real/ (synthetic Indian documents,
RTM real documents, SIDTD ID cards) and on your own documents in real_test/.

Per dataset: ROC-AUC, and recall / false-positive rate at a threshold tuned on that dataset's val split (genuine FPR
capped at 10%, as in fusion.py), plus pixel IoU on forged test images at the checkpoint's own tuned mask threshold.
Scores are cached in results/eval_model_<checkpoint>.csv.

Usage: python src/eval_model.py models/unet_crop.pt models/unet_real.pt [--n 300]"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import ROOT
from infer import load_model, predict

SPLITS_REAL, RESULTS, MAX_FPR = ROOT / "splits_real", ROOT / "results", 0.10


def sample(split: str, n: int) -> pd.DataFrame:
    """Up to n genuine + n forged images per dataset (fixed seed)."""
    df = pd.read_csv(SPLITS_REAL / f"{split}.csv", keep_default_na=False)
    df["dataset"] = df.doc_type.where(df.doc_type.isin(["rtm", "sidtd"]), "synthetic")
    df = pd.concat(g.sample(min(n, len(g)), random_state=0) for _, g in df.groupby(["dataset", "label"]))
    return df.assign(split=split)


def score(ckpt: Path, rows: pd.DataFrame) -> pd.DataFrame:
    cache = RESULTS / f"eval_model_{ckpt.stem}.csv"
    done = pd.read_csv(cache, keep_default_na=False) if cache.exists() else pd.DataFrame(columns=["path"])
    todo = rows[~rows.path.isin(done.path)]
    if len(todo):
        model = load_model(ckpt)
        thr = torch.load(ckpt, map_location="cpu", weights_only=True).get("mask_thr", 0.5)
        out = []
        for r in tqdm(todo.itertuples(), total=len(todo), desc=ckpt.stem):
            p = Path(r.path) if Path(r.path).is_absolute() else ROOT / r.path
            pred = predict(cv2.imread(str(p)), model)
            iou = np.nan
            if r.mask_path:
                mp = Path(r.mask_path) if Path(r.mask_path).is_absolute() else ROOT / r.mask_path
                gt, pm = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127, pred["mask_prob"] > thr
                iou = (gt & pm).sum() / max((gt | pm).sum(), 1)
            out.append({"path": r.path, "prob": pred["prob"], "iou": iou})
        done = pd.concat([done, pd.DataFrame(out)], ignore_index=True)
        done.to_csv(cache, index=False)
    return rows.merge(done, on="path")


def report(name: str, d: pd.DataFrame) -> list[dict]:
    rows = []
    for ds, g in d.groupby("dataset"):
        val, test = g[g.split == "val"], g[g.split == "test"]
        cands = np.unique(val.prob)
        ok = [t for t in cands if (val[val.label == 0].prob >= t).mean() <= MAX_FPR]
        t = min(ok) if ok else 1.01  # most sensitive threshold within the FPR cap
        y, p = test.label.to_numpy(), test.prob.to_numpy()
        rows.append({"model": name, "dataset": ds, "n_test": len(test), "auc": roc_auc_score(y, p),
                     "recall": (p[y == 1] >= t).mean(), "fpr": (p[y == 0] >= t).mean(),
                     "pixel_iou": test[test.label == 1].iou.astype(float).mean()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--n", type=int, default=300, help="images per dataset, label and split")
    a = ap.parse_args()
    rows = pd.concat([sample("val", a.n), sample("test", a.n)], ignore_index=True)
    own = sorted(p for p in (ROOT / "real_test").glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    table, mine = [], []
    for ck in a.checkpoints:
        table += report(ck.stem, score(ck, rows))
        if own:
            model = load_model(ck)
            mine += [{"model": ck.stem, "file": p.parent.name + "/" + p.name, "prob": predict(cv2.imread(str(p)), model)["prob"]}
                     for p in own]
    print("\n" + pd.DataFrame(table).round(3).to_string(index=False))
    if mine:
        print("\nyour documents (tamper probability):\n" + pd.DataFrame(mine).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
