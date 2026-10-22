"""Email + password accounts in a local SQLite file. Passwords are stored only as salted scrypt hashes; after
MAX_FAILS wrong passwords in a row an account is locked for LOCK_SECONDS (slows down password guessing)."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from common import ROOT

DB = ROOT / "users.db"  # git-ignored
MIN_PASSWORD = 8
MAX_FAILS, LOCK_SECONDS = 5, 300
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def _db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, salt BLOB, hash BLOB, "
                "fails INTEGER DEFAULT 0, locked_until REAL DEFAULT 0, created REAL)")
    return con


def sign_up(email: str, password: str, confirm: str, db: Path = DB) -> str | None:
    """None when the account was created, else the reason it was not."""
    email = email.strip().lower()
    if not _EMAIL.fullmatch(email):
        return "Enter a valid email address."
    if len(password) < MIN_PASSWORD:
        return f"The password must have at least {MIN_PASSWORD} characters."
    if password != confirm:
        return "The two passwords do not match."
    salt = os.urandom(16)
    with closing(_db(db)) as con, con:
        try:
            con.execute("INSERT INTO users (email, salt, hash, created) VALUES (?, ?, ?, ?)",
                        (email, salt, _hash(password, salt), time.time()))
        except sqlite3.IntegrityError:
            return "An account with this email already exists. Log in instead."
    return None


def log_in(email: str, password: str, db: Path = DB) -> str | None:
    """None when the email and password are right, else the reason the login failed."""
    email = email.strip().lower()
    with closing(_db(db)) as con, con:
        row = con.execute("SELECT salt, hash, fails, locked_until FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            _hash(password, b"0" * 16)  # same work as a real check, so timing does not reveal which emails exist
            return "Wrong email or password."
        salt, stored, fails, locked_until = row
        if locked_until > time.time():
            return f"Too many wrong passwords. Try again in {int(locked_until - time.time()) // 60 + 1} minutes."
        if hmac.compare_digest(_hash(password, salt), stored):
            con.execute("UPDATE users SET fails = 0, locked_until = 0 WHERE email = ?", (email,))
            return None
        fails += 1
        lock = time.time() + LOCK_SECONDS if fails >= MAX_FAILS else 0
        con.execute("UPDATE users SET fails = ?, locked_until = ? WHERE email = ?", (0 if lock else fails, lock, email))
    return "Wrong email or password."


if __name__ == "__main__":  # self-check on a throwaway database
    import tempfile

    db = Path(tempfile.mkdtemp()) / "users.db"
    assert sign_up("bad-email", "password123", "password123", db) == "Enter a valid email address."
    assert "at least" in sign_up("a@b.co", "short", "short", db)
    assert "do not match" in sign_up("a@b.co", "password123", "password124", db)
    assert sign_up(" A@B.co ", "password123", "password123", db) is None
    assert "already exists" in sign_up("a@b.co", "password123", "password123", db)
    assert log_in("a@b.co", "password123", db) is None and log_in("A@B.CO", "password123", db) is None
    assert log_in("nobody@b.co", "password123", db) == "Wrong email or password."
    for _ in range(MAX_FAILS):
        assert log_in("a@b.co", "wrong-pass", db) == "Wrong email or password."
    assert "Too many" in log_in("a@b.co", "password123", db)  # locked, even with the right password
    with closing(sqlite3.connect(db)) as con:
        assert b"password123" not in b"".join(r[0] for r in con.execute("SELECT hash FROM users"))
    print("auth self-check OK")
