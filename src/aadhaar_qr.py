"""Aadhaar QR code: decode it, read the holder data it carries and check UIDAI's signature.

Secure QR (2018 onwards: printed cards, PVC cards, e-Aadhaar): a base-10 number -> gzip -> fields separated by byte 255
(a 'V2' / 'V3' / 'V5' marker on newer codes, email/mobile indicator, reference id = last 4 Aadhaar digits + timestamp,
name, DOB, gender, address ...), a JPEG 2000 photo, and last a 256-byte RSA signature by UIDAI over every byte before it.
Old QR (before 2018): plain XML attributes (uid, name, gender, dob / yob), not signed.

A forger who edits the printed text but keeps the original QR is caught by the field comparison; one who rebuilds the
QR with the edited data cannot sign it, which the signature check catches (needs a UIDAI certificate in models/).
"""
from __future__ import annotations

import difflib
import gzip
import re
import zlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

CERT_DIR = Path(__file__).resolve().parent.parent / "models"  # uidai_*.cer: UIDAI Secure QR signing certificate(s)
SIG_LEN = 256
NAME_MATCH = 0.8  # ponytail: untuned guess (OCR misreads vs one-letter name edits); tune on real cards
_GENDER = {"M": "male", "F": "female", "T": "transgender"}


def _dob(s: str) -> str | None:
    """'DD/MM/YYYY', or 'YYYY' when only the year is known."""
    s = s.strip()
    if m := re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", s):
        return "/".join(m.groups())
    if m := re.fullmatch(r"(\d{4})[-/](\d{2})[-/](\d{2})", s):
        return f"{m[3]}/{m[2]}/{m[1]}"
    return s if re.fullmatch(r"\d{4}", s) else None


def _holder(name: str, dob: str, gender: str, last4: str, signed: bytes | None = None,
            signature: bytes | None = None) -> dict | None:
    """Validated holder record, or None when a field looks wrong: an unknown layout gets no check, not a false alarm."""
    name, dob, gender = name.strip(), _dob(dob), gender.strip()[:1].upper()
    if not (name and dob and gender in _GENDER and re.fullmatch(r"\d{4}", last4)):
        return None
    return {"name": name, "dob": dob, "gender": gender, "last4": last4, "signed": signed, "signature": signature}


def parse(text: str) -> dict | None:
    """Holder data from a QR payload, or None when it is not an Aadhaar QR."""
    text = text.strip()
    if text.isdigit():  # Secure QR
        n = int(text)
        try:
            data = gzip.decompress(n.to_bytes((n.bit_length() + 7) // 8, "big"))
        except (OSError, EOFError, zlib.error):
            return None
        parts = [p.decode("latin-1") for p in data.split(b"\xff", 7)[:7]]
        if re.fullmatch(r"V\d+", parts[0]):  # V2 / V3 / V5 marker; the field order after it is unchanged
            parts = parts[1:]
        if len(parts) < 5 or len(data) <= SIG_LEN:
            return None
        _, ref, name, dob, gender = parts[:5]
        return _holder(name, dob, gender, ref[:4], data[:-SIG_LEN], data[-SIG_LEN:])
    if "PrintLetterBarcodeData" in text:  # old XML QR (unsigned)
        a = dict(re.findall(r'(\w+)="([^"]*)"', text))
        return _holder(a.get("name", ""), a.get("dob") or a.get("yob", ""), a.get("gender", ""), a.get("uid", "")[-4:])
    return None


def read_qr(img_bgr: np.ndarray) -> dict | None:
    """Holder data from the Aadhaar QR on the image, or None when no readable Aadhaar QR is found."""
    import zxingcpp
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for g in (gray, cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)):  # 2x helps small, dense codes
        for r in zxingcpp.read_barcodes(g, formats=zxingcpp.BarcodeFormat.QRCode):
            if q := parse(r.text):
                return q
    return None


@lru_cache(maxsize=None)
def _uidai_keys(cert_dir: Path = CERT_DIR) -> tuple:
    from cryptography import x509
    keys = []
    for p in sorted(cert_dir.glob("uidai_*.cer")):
        b = p.read_bytes()
        load = x509.load_pem_x509_certificate if b.lstrip().startswith(b"-----") else x509.load_der_x509_certificate
        keys.append(load(b).public_key())
    return tuple(keys)


def signature_status(q: dict, cert_dir: Path = CERT_DIR) -> str:
    """'valid' / 'invalid' for a Secure QR; 'unchecked' for the unsigned old QR or when no UIDAI certificate is installed."""
    keys = _uidai_keys(cert_dir)
    if q["signature"] is None or not keys:
        return "unchecked"
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    for key in keys:  # UIDAI has rotated keys; a card signed with any installed one is accepted
        try:
            key.verify(q["signature"], q["signed"], padding.PKCS1v15(), hashes.SHA256())
            return "valid"
        except InvalidSignature:
            pass
    return "invalid"


def mismatches(q: dict, lines: list[str], number: str | None) -> list[str]:
    """Flags for printed fields that disagree with the QR. `lines`: OCR text (boxes and joined rows); `number`: the
    printed 12-digit Aadhaar number if one was read. A field the OCR did not find is not compared."""
    text, out = " ".join(lines), []
    if number and number[-4:] != q["last4"]:
        out.append("qr_number_mismatch")
    genders = {g.lower() for g in re.findall(r"\b(male|female|transgender)\b", text, re.I)}
    if genders and _GENDER[q["gender"]] not in genders:
        out.append("qr_gender_mismatch")
    dates = {"/".join(d) for d in re.findall(r"(?<!\d)(\d{2})[-/](\d{2})[-/](\d{4})(?!\d)", text)}
    if len(q["dob"]) > 4 and dates and q["dob"] not in dates:  # year-only QRs are skipped: too easy to match a wrong year
        out.append("qr_dob_mismatch")
    if lines and max(difflib.SequenceMatcher(None, q["name"].upper(), s.upper()).ratio() for s in lines) < NAME_MATCH:
        out.append("qr_name_mismatch")
    return out


if __name__ == "__main__":  # self-check: a Secure QR signed with a throwaway key round-trips, and tampering is caught
    import datetime
    import os
    import tempfile

    import qrcode
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nm = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(nm).issuer_name(nm).public_key(key.public_key())
            .serial_number(1).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    tmp = Path(tempfile.mkdtemp())
    (tmp / "uidai_test.cer").write_bytes(cert.public_bytes(serialization.Encoding.DER))

    def secure_qr(name: str, sign_name: str | None = None) -> str:
        fields = ["V2", "3", "9012" + "20240101120000000", name, "15-08-1990", "M", "S/O Test", "Pune", "", "12", "",
                  "411001", "Pune", "Maharashtra", "", "Haveli", "Pune", "4321"]
        body = lambda n: b"\xff".join(f.encode("latin-1") for f in fields[:3] + [n] + fields[4:]) + b"\xff" + os.urandom(900)
        signed = body(sign_name or name)
        sig = key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
        data = (body(name) if sign_name else signed) + sig
        return str(int.from_bytes(gzip.compress(data), "big"))

    q = parse(secure_qr("Rahul Kumar Sharma"))
    assert q and (q["name"], q["dob"], q["gender"], q["last4"]) == ("Rahul Kumar Sharma", "15/08/1990", "M", "9012"), q
    assert signature_status(q, tmp) == "valid"
    assert signature_status(q) in ("unchecked", "invalid")  # real UIDAI certs (if installed) must reject the test key
    forged = parse(secure_qr("Rohit Kumar Sharma", sign_name="Rahul Kumar Sharma"))  # QR rebuilt, old signature reused
    assert signature_status(forged, tmp) == "invalid"

    printed = ["Government of India", "Rahul Kumar Sharma", "DOB: 15/08/1990", "MALE", "2345 6789 9012"]
    assert mismatches(q, printed, "234567899012") == []
    assert mismatches(q, ["Rohit Verma", "DOB: 15/08/1991", "FEMALE"], "234567891111") == [
        "qr_number_mismatch", "qr_gender_mismatch", "qr_dob_mismatch", "qr_name_mismatch"]

    old = parse('<?xml version="1.0"?><PrintLetterBarcodeData uid="234567899012" name="Rahul Kumar Sharma" '
                'gender="M" yob="1990" co="S/O Test"/>')
    assert old and old["last4"] == "9012" and old["dob"] == "1990" and signature_status(old, tmp) == "unchecked"
    assert parse("not an aadhaar qr") is None and parse("1234567890") is None

    img = qrcode.QRCode(box_size=3, border=4)  # a dense code at about 3 px per module, as on a scanned card
    img.add_data(secure_qr("Rahul Kumar Sharma"))
    bgr = cv2.cvtColor(np.array(img.make_image().convert("RGB")), cv2.COLOR_RGB2BGR)
    assert (read_qr(bgr) or {}).get("name") == "Rahul Kumar Sharma"
    print("aadhaar_qr self-check OK")
