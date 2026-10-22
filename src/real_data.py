"""Prepare real forgery datasets for training next to the synthetic data: writes splits_real/{train,val,test}.csv
(same columns as splits/*.csv; synthetic rows included) for train_unet.py --crops --splits splits_real.

- RTM (Luo et al. 2023, github.com/DrLuo/RTM): real documents, 6000 manually manipulated + 3000 genuine, pixel masks.
  Its train list is split 90/10 into train/val; its test list is the test split.
- SIDTD (github.com/Oriolrt/SIDTD_Dataset, CC BY-SA): MIDV-2020 ID cards and passports, 1000 genuine templates and
  1222 forged ones (crop & replace, inpaint & rewrite). The forged field's box from the template annotation is the mask.
  The forgeries were saved at JPEG quality ~75 and the genuine templates at ~95: genuine templates are re-saved at 75 so
  compression history does not give the label away. Split by template number (00-79 / 80-89 / 90-99), so a forgery is
  always in the same split as the genuine card it was made from.

Download (about 3.6 GB) into <real_data>/:
  RTM/   gdown 11AHZ8ih_kDCFilGceevppcGkKR4vDJD2, unzip -> RTM/RealTextManipulation/
  SIDTD/ http://datasets.cvc.uab.es/SIDTD/templates.zip, unzip -> SIDTD/templates/
Usage: python src/real_data.py [real_data folder]   (default: ../real_data next to the project folder)"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import ROOT, SEED, SPLITS

SIDTD_QUALITY = 75  # JPEG quality of the SIDTD forgeries (luminance table starts at 8)


def rtm_rows(rtm: Path) -> dict[str, list]:
    out = {"train": [], "val": [], "test": []}
    train = (rtm / "train.txt").read_text().split()
    random.Random(SEED).shuffle(train)
    n_val = len(train) // 10
    for split, names in (("val", train[:n_val]), ("train", train[n_val:]), ("test", (rtm / "test.txt").read_text().split())):
        for n in sorted(names):
            kind = n.rsplit("_", 1)[0]
            label = int(kind != "good")
            out[split].append((str(rtm / "JPEGImages" / f"{n}.jpg"), label,
                               str(rtm / "SegmentationClass" / f"{n}.png") if label else "", "rtm",
                               kind if label else "none", f"rtm_{n}"))
    return out


def _field_boxes(ann: dict) -> dict[str, dict[str, tuple]]:
    """{template file name: {field name: (x0, y0, x1, y1)}} from a VIA annotation file."""
    out = {}
    for meta in ann["_via_img_metadata"].values():
        boxes = {}
        for r in meta["regions"]:
            s = r["shape_attributes"]
            if s["name"] == "rect":
                b = (s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"])
            elif "all_points_x" in s:
                b = (min(s["all_points_x"]), min(s["all_points_y"]), max(s["all_points_x"]), max(s["all_points_y"]))
            else:
                continue
            boxes[r["region_attributes"].get("field_name", "")] = b
        out[meta["filename"]] = boxes
    return out


def sidtd_rows(sidtd: Path) -> dict[str, list]:
    tpl, prep = sidtd / "templates", sidtd / "prepared"
    (prep / "reals").mkdir(parents=True, exist_ok=True)
    (prep / "masks").mkdir(parents=True, exist_ok=True)
    split_of = lambda idx: "train" if idx < 80 else "val" if idx < 90 else "test"
    out = {"train": [], "val": [], "test": []}
    for p in sorted((tpl / "Images" / "reals").glob("*.jpg")):  # e.g. alb_id_00.jpg
        q = prep / "reals" / p.name
        if not q.exists():
            cv2.imwrite(str(q), cv2.imread(str(p)), [cv2.IMWRITE_JPEG_QUALITY, SIDTD_QUALITY])
        out[split_of(int(p.stem.rsplit("_", 1)[1]))].append((str(q), 0, "", "sidtd", "none", f"sidtd_{p.stem}"))
    boxes = {a.stem: _field_boxes(json.loads(a.read_text(encoding="utf-8")))
             for a in (tpl / "Annotations" / "reals").glob("*.json")}  # per document type, e.g. alb_id
    skipped = 0
    for a in sorted((tpl / "Annotations" / "fakes").glob("*.json")):
        d = json.loads(a.read_text(encoding="utf-8"))
        doc, idx = d["src"].rsplit(".", 1)[0].rsplit("_", 1)  # alb_id, 00
        box = boxes.get(doc, {}).get(f"{idx}.jpg", {}).get(d["field"])
        img_path = tpl / "Images" / "fakes" / f"{d['name']}.jpg"
        if box is None or not img_path.exists():
            skipped += 1
            continue
        mask_path = prep / "masks" / f"{d['name']}.png"
        if not mask_path.exists():
            h, w = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).shape
            m = np.zeros((h, w), np.uint8)
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            m[max(0, y0):y1, max(0, x0):x1] = 255
            cv2.imwrite(str(mask_path), m)
        out[split_of(int(idx))].append((str(img_path), 1, str(mask_path), "sidtd", d["ctype"].lower(), f"sidtd_{doc}_{idx}"))
    if skipped:
        print(f"SIDTD: skipped {skipped} forgeries without a field box")
    return out


def main() -> None:
    real = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "real_data"
    parts = [rtm_rows(real / "RTM" / "RealTextManipulation"), sidtd_rows(real / "SIDTD")]
    cols = ["path", "label", "mask_path", "doc_type", "tamper_type", "base_id"]
    out_dir = ROOT / "splits_real"
    out_dir.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        syn = pd.read_csv(SPLITS / f"{split}.csv", keep_default_na=False)
        df = pd.concat([syn[cols]] + [pd.DataFrame(p[split], columns=cols) for p in parts], ignore_index=True)
        df.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"splits_real/{split}.csv: " + ", ".join(
            f"{dt} {len(g)} ({g.label.sum()} forged)" for dt, g in df.groupby("doc_type", sort=False)))


if __name__ == "__main__":
    main()
