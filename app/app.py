"""Streamlit demo: upload a document image -> image model + OCR rule checks -> fused verdict."""
from __future__ import annotations

import datetime
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ai_opinion  # noqa: E402
from auth import log_in, sign_up  # noqa: E402
from fusion import fuse, load_threshold  # noqa: E402
from infer import load_model  # noqa: E402
from pdf_input import load_image  # noqa: E402
from localize import suspicious_fields  # noqa: E402
from ocr_check import HARD_FLAGS, get_reader  # noqa: E402
from pipeline import analyze, field_mode_verdict, limit_size, mask_aadhaar  # noqa: E402
from report_pdf import build as build_report  # noqa: E402

EXPLAIN = {
    "missing_id": "No 12-digit Aadhaar number could be read from the card.",
    "invalid_aadhaar_prefix": "The Aadhaar number starts with 0 or 1, which a real Aadhaar number never does.",
    "aadhaar_checksum_fail": "The Aadhaar number fails the Verhoeff checksum, so it cannot be a valid number.",
    "impossible_dob": "A printed date is not a real calendar date (for example 29/02 in a year that is not a leap year).",
    "invalid_dob": "The date of birth is missing, is not a real DD/MM/YYYY date, or gives an age outside 0-120.",
    "invalid_gender": "The gender is not one of Male, Female or Transgender.",
    "missing_qr": "No QR code was found on the card.",
    "qr_name_mismatch": "The printed name is different from the name stored in the card's QR code.",
    "qr_dob_mismatch": "The printed date of birth is different from the one stored in the card's QR code.",
    "qr_gender_mismatch": "The printed gender is different from the one stored in the card's QR code.",
    "qr_number_mismatch": "The last 4 digits of the printed Aadhaar number differ from the ones stored in the QR code.",
    "qr_signature_invalid": "The QR code's UIDAI digital signature is not valid, so the QR was made or changed by "
                            "someone other than UIDAI.",
    "invalid_pan_format": "The PAN does not match the required format of five letters, four digits and a letter.",
    "invalid_pan_type": "The 4th PAN character is not a valid holder-type code (P, C, H, F, A, T, B, L, J or G).",
    "pan_surname_mismatch": "For an individual PAN, the 5th character must be the first letter of the surname, "
                            "but it does not match the printed name.",
    "missing_header": "The 'INCOME TAX DEPARTMENT' header could not be found.",
    "marks_exceed_max": "A subject's marks obtained are higher than its maximum marks.",
    "total_mismatch": "The printed total does not equal the sum of the subject marks.",
    "percentage_mismatch": "The printed percentage does not match the total divided by the maximum marks.",
    "result_mismatch": "The printed PASS/FAIL result contradicts the rule that a mark below 33 in any subject is a fail.",
    "font_inconsistency": "Some text lines have clearly different font heights from their neighbours, "
                          "which can happen when text has been overwritten.",
    "low_ocr_confidence": "The text is hard to read (low OCR confidence), so the rule checks may be unreliable.",
    "model_score_high": "The image model sees pixel-level patterns typical of edited regions.",
    "suspicious_fields": "Some text has letter edges crisper than the text around it, which can happen when text is "
                         "re-typed. This is only a hint: in tests it appeared on genuine documents as often as on forged "
                         "ones, so it does not change the verdict. Look at the fields listed under the heatmap yourself.",
}
HEATMAP_MIN, HEATMAP_ALPHA = 0.3, 0.6
# The U-Net was trained only on the generated synthetic documents; on real scans it reports a high tamper probability for
# genuine documents too, so it stays off. Set USE_UNET = True to bring back its heatmap and fused score.
USE_UNET, USE_TUNED, MANUAL_THR = False, True, 0.5

st.set_page_config(page_title="FORGETRACE", page_icon=":material/fact_check:", layout="wide", initial_sidebar_state="collapsed")
st.html(f"<style>{(ROOT / 'app' / 'style.css').read_text(encoding='utf-8')}</style>")


# ---------------------------------------------------------------- cached resources and helpers
@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_resources():
    get_reader()  # warm up the OCR reader so the first upload is not slow
    if not USE_UNET:  # skip the ~100 MB U-Net: it is unused and the free host has little RAM
        return None, {}
    return load_model(), load_threshold()


@st.cache_data(max_entries=16, show_spinner=False)
def run_pipeline(img_bytes: bytes) -> dict:
    """Heavy part (model + OCR), see src/pipeline.py."""
    return analyze(img_bytes, load_resources()[0] if USE_UNET else None)


_t = np.linspace(0, 1, 256, dtype=np.float32)[:, None]  # orange -> pale amber lookup table (BGR), no blue / purple
ORANGE_LUT = (np.array([10, 70, 255], np.float32) * (1 - _t) + np.array([190, 235, 255], np.float32) * _t).astype(np.uint8)[:, None, :]


def heatmap_overlay(img_bgr: np.ndarray, mask_prob: np.ndarray) -> np.ndarray:
    """Orange heatmap blended at alpha 0.45, only where mask_prob > 0.3. Returns RGB."""
    heat = cv2.applyColorMap((np.clip(mask_prob.astype(np.float32), 0, 1) * 255).astype(np.uint8), ORANGE_LUT)
    on = (mask_prob > HEATMAP_MIN)[..., None]
    blended = cv2.addWeighted(img_bgr, 1 - HEATMAP_ALPHA, heat, HEATMAP_ALPHA, 0)
    return cv2.cvtColor(np.where(on, blended, img_bgr).astype(np.uint8), cv2.COLOR_BGR2RGB)


def flag_label(flag: str) -> str:
    return flag.replace("_", " ")


# ---------------------------------------------------------------- settings (no sidebar)
use_unet, use_tuned, manual_thr = USE_UNET, USE_TUNED, MANUAL_THR

# ---------------------------------------------------------------- main page
st.html("""
<div class="hero">
  <div class="nav"><b>FORGE<i>TRACE</i></b><span class="sep"></span><span>Document forgery check</span></div>
  <h1>Forge<span style="-webkit-text-fill-color:#FF6A1A">trace</span></h1>
  <div class="vs">Authentic vs forged</div>
  <p>Upload an Aadhaar card, PAN card or marksheet. Every text field is compared with the text around it and the
  printed numbers are checked against their own rules.</p>
</div>
""")

# ---------------------------------------------------------------- account (src/auth.py)
if "user" not in st.session_state:
    log_tab, sign_tab = st.tabs(["Log in", "Sign up"])
    with log_tab, st.form("login"):
        email = st.text_input("Email", key="li_email", autocomplete="email")
        password = st.text_input("Password", type="password", key="li_pw", autocomplete="current-password")
        if st.form_submit_button("Log in", type="primary"):
            err = log_in(email, password)
            if err:
                st.error(err)
            else:
                st.session_state.user = email.strip().lower()
                st.rerun()
    with sign_tab, st.form("signup"):
        email = st.text_input("Email", key="su_email", autocomplete="email")
        password = st.text_input("Password (at least 8 characters)", type="password", key="su_pw",
                                 autocomplete="new-password")
        confirm = st.text_input("Confirm password", type="password", key="su_pw2", autocomplete="new-password")
        if st.form_submit_button("Create account", type="primary"):
            err = sign_up(email, password, confirm)
            if err:
                st.error(err)
            else:
                st.session_state.user = email.strip().lower()
                st.rerun()
    st.stop()

with st.container(horizontal=True, vertical_alignment="center"):
    st.caption(f"Signed in as **{st.session_state.user}**")
    if st.button("Log out", icon=":material/logout:"):
        del st.session_state["user"]
        st.rerun()

source = st.segmented_control("Image source", ["Upload", "Camera"], default="Upload", label_visibility="collapsed",
                              selection_mode="single", key="source") or "Upload"
if source == "Camera":
    uploaded = st.camera_input("Take a photo of the document", resolution="1080p")
    st.caption("Lay the document flat in good light, fill the frame and hold steady. Tap the camera-switch icon "
               "to use the rear camera.")
else:
    uploaded = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png", "pdf"])
    st.caption("JPG, PNG or a single-page PDF.")
st.caption("Uploaded images are processed in memory and never saved to disk.")

if uploaded is not None:
    img_bytes, truth = uploaded.getvalue(), None
else:
    st.info("Upload a JPG, PNG or single-page PDF, or take a photo, of an Aadhaar card, PAN card or marksheet.",
            icon=":material/upload_file:")
    st.stop()

model, cfg = load_resources()
if model is not None and cfg.get("model") and cfg["model"] != getattr(model, "file", cfg["model"]):
    st.warning(f"The saved thresholds were tuned for `{cfg['model']}`, but `{model.file}` is loaded. "
               "Re-run `python src/fusion.py` to re-tune them.", icon=":material/warning:")

with st.spinner("Analysing the document..."):
    res = run_pipeline(img_bytes)
if "error" in res:
    st.error(res["error"])
    st.stop()
if "rejected" in res:
    st.error(f"**Image rejected.** {res['rejected']}", icon=":material/block:")
    st.caption("Accepted: a clear photo or scan of an Aadhaar card or letter, a PAN card, or a school marksheet.")
    st.stop()

img = limit_size(load_image(img_bytes)[0])  # run_pipeline already succeeded, so this cannot fail
ocr, prob = res["ocr"], res["model_prob"]
suspicious = suspicious_fields(res["fields"])
if use_unet:
    verdict = fuse(prob, ocr, cfg if use_tuned else manual_thr)
    threshold = (cfg["per_doc_type"].get(ocr["doc_type"], cfg["threshold"]) if use_tuned else manual_thr)
    tampered = verdict["verdict"] == "TAMPERED"
else:
    verdict = field_mode_verdict(res)
    threshold, tampered = float("nan"), verdict["verdict"] == "TAMPERED"
# The checks can show forgery but cannot prove a document genuine, so the other verdict says only that nothing was found
label = "TAMPERED" if tampered else "NO FORGERY FOUND"

# verdict + scores
with st.container(border=True):
    top = st.container(horizontal=True, vertical_alignment="center")
    with top:
        if tampered:
            st.badge(label, icon=":material/gpp_bad:", color="red")
        else:
            st.badge(label, icon=":material/verified:", color="green")
        st.markdown(f"Detected document type: **{ocr['doc_type']}**")
        if truth:
            st.markdown(f"Ground truth: **{truth}**")
    c = st.columns(4 if use_unet else 2)
    c[0].metric("OCR score", f"{ocr['ocr_score']:.2f}",
                help="Rule-check score from the text read by OCR. 1.00 if any hard rule failed (impossible date, "
                     "Aadhaar checksum, PAN format, marksheet totals). Otherwise the number of soft warnings divided by 3 "
                     "(0.33 = one warning). It adds context only: the TAMPERED verdict comes from hard rule failures or "
                     "suspicious text fields, not from this score.")
    c[1].metric("Suspicious text fields", len(suspicious),
                help="Text fields whose edges are crisper than the text around them, plus any field a failed rule points "
                     "at. A hint for you to check, not evidence: only failed rules give TAMPERED.")
    if use_unet:
        c[2].metric("Model probability", f"{prob:.3f}")
        c[3].metric("Final / threshold", f"{verdict['final_score']:.2f} / {threshold:.2f}")
    notes = (["An OCR hard failure overrides the threshold and always gives TAMPERED."] if ocr["hard_fail"] else []) + (
        [{"valid": "Aadhaar QR: UIDAI signature verified, and the printed details were checked against it.",
          "invalid": "Aadhaar QR: the UIDAI signature check failed.",
          "unchecked": "Aadhaar QR read and the printed details were checked against it. Its UIDAI signature "
                       "was not verified (old unsigned QR, or no UIDAI certificate in models/)."}[ocr["qr"]]]
        if ocr.get("qr") else [])
    for n in notes:
        st.caption(n)

# reasons
st.subheader("Why")
reasons = verdict["reasons"]
if reasons:
    for r in reasons:
        icon = ":material/error:" if r in HARD_FLAGS else ":material/warning:"
        (st.error if r in HARD_FLAGS else st.warning)(f"**{flag_label(r).capitalize()}.** {EXPLAIN.get(r, r)}", icon=icon)
else:
    st.success("No rule violations were found and the image model sees nothing suspicious.", icon=":material/check_circle:")

# images
img_rgb, heat_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), heatmap_overlay(img, res["mask_prob"] if use_unet else res["field_map"])
table = None
left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(img_rgb, alt="The uploaded document", width="stretch")
with right:
    st.subheader("Tamper heatmap")
    st.image(heat_rgb, width="stretch", alt="The document with the suspected edited regions highlighted in colour")
    st.caption("Text fields whose ink and background look different from the text around them (re-drawn text is "
               "usually crisper than the surrounding scan). Redder = stronger. Each field is compared only with its "
               "neighbours in the same image, so it does not depend on the upload's resolution or compression."
               if not use_unet else "Output of the synthetic-trained U-Net (reliable only on the generated documents).")
    if not use_unet and suspicious:
        lines: dict[tuple, dict] = {}  # one row per printed line: its flagged words are merged
        for f in suspicious:
            text = ai_opinion.mask_numbers([f["text"]])[0]  # a flagged ID number must not show in full
            row = lines.setdefault((text, f["bbox"][1] // 40), {"Text": text, "Flagged words": 0, "Strength": 0.0,
                                                                "Why": f["reason"]})
            row["Flagged words"] += 1
            if f["score"] > row["Strength"]:
                row["Strength"], row["Why"] = f["score"], f["reason"]
        table = pd.DataFrame(sorted(lines.values(), key=lambda r: -r["Strength"])[:12])
        table["Strength"] = table["Strength"].map(lambda v: "rule failed" if v >= 99 else f"{v:.1f}")
        st.dataframe(table, hide_index=True, width="stretch", alt="Text lines flagged as possibly re-typed")
        st.caption("Strength: how far the text stands out from its neighbours (4.0 or more is flagged). "
                   "Why: `edge` = letter edges much crisper than the surrounding text; other entries are failed rules.")

# AI analysis and opinion (src/ai_opinion.py): one request per document, findings only, never the image
st.subheader("AI analysis and opinion")
ai_key = "ai_" + hashlib.sha1(img_bytes).hexdigest()
ai_text = st.session_state.get(ai_key)
if not ai_opinion.available():
    st.caption("Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` to get Gemini's analysis of these findings.")
elif ai_text is None:
    st.caption("Sends the verdict, flags, scores and extracted text to Google's Gemini API, with Aadhaar, phone and "
               "other long numbers masked. The document image is not sent.")
    if st.button("Get AI opinion", icon=":material/psychology:"):
        with st.spinner("Asking Gemini..."):
            try:
                ai_text = st.session_state[ai_key] = ai_opinion.opinion(
                    ai_opinion.findings({**verdict, "verdict": label}, ocr, suspicious, EXPLAIN))
            except RuntimeError as e:
                st.error(str(e), icon=":material/cloud_off:")
if ai_text:
    with st.container(border=True):
        st.markdown(ai_text)
    st.caption("Advisory only: written by Gemini from the findings above, without seeing the document. "
               "The verdict comes from FORGETRACE's checks.")

with st.expander("Extracted text"):
    st.caption("The first 8 digits of any Aadhaar number are hidden.")
    st.code("\n".join(mask_aadhaar(ocr["text"])) or "(no text found)", language=None)

# PDF report (src/report_pdf.py): numbers masked more strongly than on screen, because a PDF gets shared
scores = {"OCR score": f"{ocr['ocr_score']:.2f}", "Suspicious text fields": str(len(suspicious))}
if use_unet:
    scores |= {"Model probability": f"{prob:.3f}", "Final / threshold": f"{verdict['final_score']:.2f} / {threshold:.2f}"}
pdf = build_report(verdict=label, doc_type=ocr["doc_type"], scores=scores, notes=notes, ai_text=ai_text,
                   reasons=[(flag_label(r).capitalize(), EXPLAIN.get(r, r), r in HARD_FLAGS) for r in reasons],
                   fields=[] if table is None else table.to_dict("records"),
                   text_lines=ai_opinion.mask_numbers(ocr["text"]), image_rgb=img_rgb, heat_rgb=heat_rgb,
                   user=st.session_state.user)
st.download_button("Download PDF report", data=pdf, mime="application/pdf", icon=":material/download:", type="primary",
                   file_name=f"forgetrace_{ocr['doc_type']}_{datetime.datetime.now():%Y%m%d_%H%M}.pdf")
