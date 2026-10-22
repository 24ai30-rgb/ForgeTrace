"""Evaluate the app on your own real documents: <folder>/genuine/* and <folder>/tampered/* (JPG / PNG / single-page PDF).
Runs the exact app pipeline (src/pipeline.py) on this computer; nothing is uploaded. Writes <folder>/results.csv inside
the (git-ignored) folder, because file names can be personal.

Usage: python src/eval_real.py [folder]   (default: real_test/)"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from common import ROOT
from pipeline import analyze, field_mode_verdict

EXTS = {".jpg", ".jpeg", ".png", ".pdf"}


def run(folder: Path) -> pd.DataFrame:
    rows = []
    for truth in ("genuine", "tampered"):
        for p in sorted((folder / truth).glob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            print(f"{truth}/{p.name} ...", flush=True)
            res = analyze(p.read_bytes())
            if "error" in res or "rejected" in res:
                rows.append({"file": p.name, "truth": truth, "verdict": "REJECTED", "why": res.get("error") or res["rejected"]})
                continue
            v = field_mode_verdict(res)
            rows.append({"file": p.name, "truth": truth, "doc_type": res["ocr"]["doc_type"], "verdict": v["verdict"],
                         "why": ", ".join(v["reasons"]) or "-", "suspicious_fields": len(v["suspicious"]),
                         "qr": res["ocr"]["qr"]})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print("\n" + df.to_string(index=False))
    ok = df[df.verdict != "REJECTED"]
    g, t = ok[ok.truth == "genuine"], ok[ok.truth == "tampered"]
    print(f"\nrejected: {len(df) - len(ok)} of {len(df)}")
    print(f"genuine marked TAMPERED (false alarms): {(g.verdict == 'TAMPERED').sum()} of {len(g)}")
    print(f"tampered caught: {(t.verdict == 'TAMPERED').sum()} of {len(t)}")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "real_test"
    df = run(folder)
    if df.empty:
        sys.exit(f"No documents found. Put files in {folder / 'genuine'} and {folder / 'tampered'}.")
    df.to_csv(folder / "results.csv", index=False)
    report(df)
