"""The app's analysis without the UI: image / PDF bytes -> rejection, or OCR + field + model results and a verdict.
Shared by app/app.py and src/eval_real.py, so the real-document evaluation measures exactly what the app does."""
from __future__ import annotations

import re

import cv2
import numpy as np

from localize import anomaly_map, field_anomalies, suspicious_fields
from ocr_check import ocr_check
from pdf_input import load_image
from validate import check_document, check_image

MAX_SIDE = 2000  # px; phone photos are 12 MP+, which exhausts the free host's RAM during OCR


def mask_aadhaar(lines: list[str]) -> list[str]:
    """Hide the first 8 digits of any 12-digit number (also when OCR split it into three 4-digit lines)."""
    pat = re.compile(r"(?<!\d)(\d{4})(\s?)(\d{4})(\s?)(\d{4})(?!\d)")
    out = [pat.sub(lambda m: f"XXXX{m.group(2)}XXXX{m.group(4)}{m.group(5)}", ln) for ln in lines]
    for i in range(len(out) - 2):
        if all(re.fullmatch(r"\d{4}", out[j]) for j in range(i, i + 3)):
            out[i], out[i + 1] = "XXXX", "XXXX"
    return out


def limit_size(img: np.ndarray) -> np.ndarray:
    scale = MAX_SIDE / max(img.shape[:2])
    return img if scale >= 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def analyze(img_bytes: bytes, model=None) -> dict:
    """{'error': ...} or {'rejected': ...}, else model / OCR / field results. The U-Net runs only when `model` is given.
    Works on in-memory bytes only; nothing is written to disk."""
    img, err = load_image(img_bytes)
    if img is None:
        return {"error": err}
    img = limit_size(img)
    reason = check_image(img)
    if reason:
        return {"rejected": reason}
    ocr = ocr_check(img)
    reason = check_document(ocr)
    if reason:
        return {"rejected": reason}
    pred = {"prob": 0.0, "mask_prob": np.zeros(img.shape[:2], np.float16)}
    if model is not None:
        from infer import predict
        pred = predict(img, model)
    fields = field_anomalies(img) + [  # fields a failed rule points at (e.g. an impossible date) are highlighted too
        {"text": r["text"], "bbox": r["bbox"], "score": 99.0, "conf": r["conf"], "reason": r["flag"].replace("_", " ")}
        for r in ocr["rule_boxes"]]
    return {"model_prob": pred["prob"], "mask_prob": pred["mask_prob"].astype(np.float16), "ocr": ocr,
            "fields": fields, "field_map": anomaly_map(img.shape, fields).astype(np.float16)}


def field_mode_verdict(res: dict) -> dict:
    """Verdict without the U-Net: TAMPERED only when a hard OCR / QR rule fails (0% of genuine test documents).
    Suspicious text fields (localize.py) are reported as a weak hint but do not decide: on 400 real RTM documents they
    appeared on more genuine pages (21%) than forged ones (12.5%)."""
    ocr, suspicious = res["ocr"], suspicious_fields(res["fields"])
    return {"verdict": "TAMPERED" if ocr["hard_fail"] else "GENUINE",
            "final_score": float(ocr["ocr_score"]),
            "reasons": list(ocr["flags"]) + (["suspicious_fields"] if suspicious else []), "suspicious": suspicious}
