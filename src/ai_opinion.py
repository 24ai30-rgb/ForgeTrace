"""One-shot AI analysis and opinion from Gemini on FORGETRACE's findings (a report, not a chatbot).

Only the findings are sent: document type, verdict, raised flags with their meaning, scores, suspicious fields and the
OCR text with Aadhaar / phone / other long numbers masked. The document image never leaves the app.
Configuration: GEMINI_API_KEY (required) and GEMINI_MODEL (optional), as environment variables or as root-level
entries in .streamlit/secrets.toml (Streamlit exposes those as environment variables)."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from ocr_check import HARD_FLAGS
from pipeline import mask_aadhaar

DEFAULT_MODEL = "gemini-3.6-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
INSTRUCTIONS = """You are a document-forensics assistant reviewing the output of FORGETRACE, an automated forgery
checker for Indian documents (Aadhaar card, PAN card, school marksheet). You cannot see the document, only the findings
in the user message. Write a short analysis and opinion for the person verifying it, with these headed sections:
1. Opinion: one or two sentences, saying how confident the evidence allows anyone to be.
2. Findings: what each finding means and how strong it is as evidence of forgery. Failed hard rules (decisive = true)
   are strong evidence. Suspicious text fields and OCR warnings are weak, and are often caused by image quality.
3. Innocent explanations to rule out.
4. Next steps a person can take, such as scanning the Aadhaar QR code with UIDAI's official app, verifying the PAN on
   the income tax e-filing portal, or checking the marksheet on DigiLocker or the board's website.
Never claim certainty, never invent findings, and do not repeat personal details. Plain English, under 250 words."""


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def mask_numbers(lines: list[str]) -> list[str]:
    """Aadhaar numbers (also split over lines) and any other number of 8+ digits keep only their last 4 digits."""
    long_num = re.compile(r"\d(?:[ -]?\d){7,}")
    return [long_num.sub(lambda m: re.sub(r"\d", "X", m.group()[:-4]) + m.group()[-4:], ln) for ln in mask_aadhaar(lines)]


def findings(verdict: dict, ocr: dict, suspicious: list[dict], explain: dict[str, str]) -> dict:
    """The (masked) facts the AI is given."""
    return {
        "document_type": ocr["doc_type"],
        "automated_verdict": verdict["verdict"],
        "findings": [{"flag": r, "meaning": explain.get(r, r), "decisive": r in HARD_FLAGS} for r in verdict["reasons"]],
        "ocr_rule_score_0_to_1": round(float(ocr["ocr_score"]), 2),
        "aadhaar_qr": {None: "not read", "unchecked": "read, compared with the printed text, signature not verified",
                       "valid": "read, UIDAI signature valid", "invalid": "UIDAI signature invalid"}.get(ocr.get("qr")),
        "suspicious_text_fields": [{"text": mask_numbers([f["text"]])[0], "strength_z": round(float(f["score"]), 1),
                                    "why": f.get("reason", "")} for f in suspicious[:10]],
        "extracted_text": mask_numbers(ocr["text"])[:80],
    }


def opinion(facts: dict, timeout: float = 60) -> str:
    """Gemini's analysis as Markdown. Raises RuntimeError with a readable message on any failure."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key. Add GEMINI_API_KEY to .streamlit/secrets.toml or the environment.")
    body = {"systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(facts, ensure_ascii=False, indent=1)}]}]}
    req = urllib.request.Request(URL.format(model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)),
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Gemini returned HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Could not reach Gemini: {e}") from None
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise RuntimeError("Gemini returned no text (the response was empty or blocked).")
    return text


if __name__ == "__main__":  # self-check, no network
    assert mask_numbers(["2345 6789 0123", "Mobile: 9876543210", "DOB: 01/01/1990", "PIN 400001"]) == [
        "XXXX XXXX 0123", "Mobile: XXXXXX3210", "DOB: 01/01/1990", "PIN 400001"]
    assert mask_numbers(["1234", "5678", "9012"]) == ["XXXX", "XXXX", "9012"]
    f = findings({"verdict": "TAMPERED", "reasons": ["aadhaar_checksum_fail", "suspicious_fields"]},
                 {"doc_type": "aadhaar", "ocr_score": 1.0, "qr": None, "text": ["Your Aadhaar No.: 2345 6789 0123"]},
                 [{"text": "2345 6789 0123", "score": 7.43, "reason": "edge"}], {"aadhaar_checksum_fail": "checksum"})
    assert f["findings"][0]["decisive"] and not f["findings"][1]["decisive"]
    assert "2345" not in json.dumps(f) and "0123" in json.dumps(f)
    os.environ.pop("GEMINI_API_KEY", None)
    try:
        opinion(f)
        raise AssertionError("expected RuntimeError without a key")
    except RuntimeError as e:
        assert "GEMINI_API_KEY" in str(e)
    print("ai_opinion self-check OK")
