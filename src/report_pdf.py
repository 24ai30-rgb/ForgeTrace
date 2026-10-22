"""PDF report of one analysis: HTML laid out by PyMuPDF's Story over as many A4 pages as needed."""
from __future__ import annotations

import datetime
import html
import io

import cv2
import numpy as np
import pymupdf

CSS = """
* {font-family: sans-serif; font-size: 10pt; color: #222;}
h1 {font-size: 20pt; margin: 0 0 2pt 0;} h2 {font-size: 13pt; margin: 14pt 0 4pt 0; color: #C2410C;}
.meta {color: #666; font-size: 9pt;} .verdict {font-size: 16pt; font-weight: bold;}
.bad {color: #B91C1C;} .good {color: #15803D;} .warn {color: #B45309;}
td, th {padding: 3pt 6pt; vertical-align: top; text-align: left;} th {background-color: #F3E8DF;}
.small {font-size: 8pt; color: #555;} pre {font-family: monospace; font-size: 8pt;}
"""


def _jpeg(rgb: np.ndarray, max_side: int = 900) -> bytes:
    s = min(1.0, max_side / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    return cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def build(*, verdict: str, doc_type: str, reasons: list[tuple[str, str, bool]], scores: dict[str, str],
          fields: list[dict], notes: list[str], ai_text: str | None, text_lines: list[str],
          image_rgb: np.ndarray, heat_rgb: np.ndarray, user: str) -> bytes:
    """reasons: (label, explanation, decisive). fields: dicts with Text / Strength / Why. Returns the PDF bytes.
    Pass already-masked text: this function prints what it is given."""
    e = html.escape
    arch = pymupdf.Archive()
    arch.add((_jpeg(image_rgb), "original.jpg"))
    arch.add((_jpeg(heat_rgb), "heatmap.jpg"))
    cls = "bad" if verdict == "TAMPERED" else "good"
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    parts = [f"<h1>FORGETRACE analysis report</h1><p class='meta'>{e(now)} &middot; requested by {e(user)}</p>",
             f"<p class='verdict {cls}'>{e(verdict)}</p><p>Detected document type: <b>{e(doc_type)}</b></p>",
             "<table>" + "".join(f"<tr><td>{e(k)}</td><td><b>{e(v)}</b></td></tr>" for k, v in scores.items()) + "</table>"]
    parts.append("<h2>Why</h2>")
    parts += [f"<p><b class='{'bad' if hard else 'warn'}'>{e(label)}.</b> {e(text)}</p>" for label, text, hard in reasons] or [
        "<p class='good'>No rule violations were found.</p>"]
    parts += [f"<p class='small'>{e(n)}</p>" for n in notes]
    parts.append("<h2>Document and suspected regions</h2><table><tr>"
                 "<td><img src='original.jpg' width='235'/><p class='small'>Original</p></td>"
                 "<td><img src='heatmap.jpg' width='235'/><p class='small'>Suspected edited regions</p></td></tr></table>")
    if fields:
        parts.append("<h2>Flagged text</h2><table><tr><th>Text</th><th>Strength</th><th>Why</th></tr>" + "".join(
            f"<tr><td>{e(str(f['Text']))}</td><td>{e(str(f['Strength']))}</td><td>{e(str(f['Why']))}</td></tr>"
            for f in fields) + "</table>")
    if ai_text:
        parts.append("<h2>AI analysis and opinion (Gemini)</h2><p class='small'>Advisory only. Written by Gemini from the "
                     "findings above, without seeing the document. The verdict comes from FORGETRACE's checks.</p>")
        parts += [f"<p>{e(ln.replace('**', '').lstrip('#').strip())}</p>" for ln in ai_text.splitlines() if ln.strip()]
    parts.append("<h2>Extracted text</h2><pre>" + e("\n".join(text_lines) or "(no text found)") + "</pre>")
    parts.append("<p class='small'>FORGETRACE is a research prototype trained largely on synthetic documents. It must not "
                 "be used on its own to accept or reject real identity documents: confirm with the issuing authority "
                 "(UIDAI QR scan, income tax portal, DigiLocker).</p>")

    story = pymupdf.Story(html="".join(parts), user_css=CSS, archive=arch)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    page = pymupdf.paper_rect("a4")
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(page + (40, 40, -40, -40))
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


if __name__ == "__main__":  # self-check: a long report spills onto a second page and keeps its text
    img = np.full((400, 640, 3), 230, np.uint8)
    pdf = build(verdict="TAMPERED", doc_type="aadhaar", reasons=[("Aadhaar checksum fail", "Fails the Verhoeff check.", True)],
                scores={"OCR score": "1.00", "Suspicious text fields": "1"},
                fields=[{"Text": "Name <b>&", "Strength": "7.4", "Why": "edge"}], notes=["Aadhaar QR: not read."],
                ai_text="**Opinion**\nLikely forged.\n" * 40, text_lines=["line"] * 120, image_rgb=img, heat_rgb=img, user="a@b.co")
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    text = "".join(p.get_text() for p in doc)
    assert pdf[:5] == b"%PDF-" and doc.page_count >= 2, doc.page_count
    assert "TAMPERED" in text and "Name <b>&" in text and "Gemini" in text and "a@b.co" in text
    assert len(doc[0].get_image_info()) == 2  # placements (identical images are stored once)
    print(f"report_pdf self-check OK ({doc.page_count} pages)")
