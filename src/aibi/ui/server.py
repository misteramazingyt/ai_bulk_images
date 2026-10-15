"""Swipe-to-code web app for human coders (phone-first).

`aibi code-ui` serves a single-page app over HTTP on the local network. All
coders see the same seeded sample of images, each coder in their own
shuffled order. Coders see only the image (no caption, platform, or engagement)
to keep coding blind (CODEBOOK.md "Unit of analysis").

Every decision is appended to data/coding/<coder>.jsonl (latest event per
item wins, so undo is just another event). `aibi ui-export` turns a log into
the standard coding sheet consumed by `aibi import-codes` and `aibi reliability`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..codebook import Codebook
from ..coders.human import AFFECT_COLS
from ..models import utcnow
from ..safety import apply_safety_flag
from ..storage import Store

log = logging.getLogger("aibi.ui")
INDEX = Path(__file__).with_name("index.html")
CODER_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
FLAG_REASONS = {"possible_minor_sexualization", "sexualized_real_person", "other_harmful"}


def item_id(record_id: str) -> str:
    """Opaque id so the client never sees platform-revealing record ids."""
    return hashlib.sha256(record_id.encode()).hexdigest()[:16]


def select_codes(cb: Codebook, spec: str) -> list[str]:
    if spec == "all":
        return list(cb.codes)
    if spec == "visual":
        return [c.id for c in cb.visual_motifs]
    if spec == "image":  # visual motifs + techniques an image can carry without text
        return [c.id for c in cb.visual_motifs] + [c.id for c in cb.techniques if c.modality != "text"]
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [i for i in ids if i not in cb.codes]
    if unknown:
        raise ValueError(f"unknown code ids: {unknown}")
    return ids


class CodingSession:
    def __init__(self, store: Store, cb: Codebook, codes: list[str], sample: int | None = None, seed: int = 0,
                 token: str | None = None):
        self.store, self.cb, self.codes = store, cb, codes
        self.token = token or secrets.token_urlsafe(12)
        self.lock = threading.Lock()
        self.log_dir = store.root / "coding"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        eligible = sorted((r for r in store.iter_records() if not r.exclusion and r.image.local_path),
                          key=lambda r: hashlib.sha256(f"{seed}:{r.record_id}".encode()).hexdigest())
        self.records = {item_id(r.record_id): r for r in (eligible[:sample] if sample else eligible)}
        self.hidden: set[str] = set()

    # --- coder logs -----------------------------------------------------
    def _log_path(self, coder: str) -> Path:
        return self.log_dir / f"{coder}.jsonl"

    def append(self, coder: str, event: dict) -> None:
        event = {"t": utcnow().isoformat(), **event}
        with self.lock, self._log_path(coder).open("a") as f:
            f.write(json.dumps(event) + "\n")

    def state(self, coder: str) -> dict:
        return replay(self._log_path(coder))

    # --- API payloads ---------------------------------------------------
    def session_payload(self, coder: str) -> dict:
        order = sorted((i for i in self.records if i not in self.hidden),
                       key=lambda i: hashlib.sha256(f"{coder}:{i}".encode()).hexdigest())
        st = self.state(coder)
        codes = [{"id": c.id, "name": c.name.replace("_", " "), "layer": c.layer, "modality": c.modality,
                  "definition": c.definition, "include": c.include, "exclude": c.exclude}
                 for c in (self.cb.codes[i] for i in self.codes)]
        return {
            "codebook_version": self.cb.version,
            "items": order,
            "codes": codes,
            "emotions": self.cb.emotions,
            "affect_anchors": {k: v.get("anchors", {}) for k, v in self.cb.affect.get("dimensions", {}).items()},
            "done_motifs": [f"{i}|{c}" for (i, c), v in st["motifs"].items() if v is not None],
            "done_affect": [i for i, v in st["affect"].items() if v is not None],
        }

    def flag(self, coder: str, iid: str, reason: str) -> None:
        rec = self.records.get(iid)
        if rec is None:
            return
        self.append(coder, {"kind": "flag", "item": iid, "reason": reason})
        self.hidden.add(iid)
        with self.lock:
            recs = self.store.load(rec.source.platform)
            stored = recs.get(rec.record_id)
            if stored is not None:
                if reason == "other_harmful":
                    stored.exclusion = "flagged_harmful_pending_review"
                else:
                    apply_safety_flag(stored, reason, source=f"human:{coder}")
                self.store.write(rec.source.platform, recs.values())
        log.warning("item %s flagged by %s: %s", rec.record_id, coder, reason)


def replay(path: Path) -> dict:
    """Latest-event-wins state from a coder log."""
    motifs: dict[tuple[str, str], object] = {}
    affect: dict[str, dict | None] = {}
    flags: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["kind"] == "motif":
                motifs[(e["item"], e["code"])] = e["value"]
            elif e["kind"] == "affect":
                affect[e["item"]] = e.get("value")
            elif e["kind"] == "flag":
                flags[e["item"]] = e["reason"]
    return {"motifs": motifs, "affect": affect, "flags": flags}


def export_coder(session_store: Store, cb: Codebook, coder: str, out: str | Path) -> int:
    """Write a coder's log as a standard coding sheet (see coders/human.py)."""
    st = replay(session_store.root / "coding" / f"{coder}.jsonl")
    by_iid = {item_id(r.record_id): r for r in session_store.iter_records()}
    items = sorted({i for i, _ in st["motifs"]} | set(st["affect"]))
    cols = ["record_id", "post_url", "local_path", *cb.codes.keys(), *AFFECT_COLS, "note"]
    n = 0
    with Path(out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for iid in items:
            rec = by_iid.get(iid)
            if rec is None or iid in st["flags"]:
                continue
            row = {"record_id": rec.record_id, "post_url": rec.source.post_url or "", "local_path": ""}
            unsure = []
            for (i, code), v in st["motifs"].items():
                if i != iid or v is None:
                    continue
                if v == "unsure":
                    unsure.append(code)
                else:
                    row[code] = str(int(v))
            a = st["affect"].get(iid)
            if a:
                for k in AFFECT_COLS:
                    if a.get(k) is not None:
                        row[k] = a[k]
            if unsure:
                row["note"] = "unsure:" + ";".join(sorted(unsure))
            w.writerow(row)
            n += 1
    return n


def make_handler(session: CodingSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "aibi-code-ui"

        def log_message(self, fmt, *args):  # quieter default logging
            log.debug("%s - " + fmt, self.address_string(), *args)

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

        def _authorized(self, q: dict) -> bool:
            tok = self.headers.get("X-Token") or (q.get("t") or [""])[0]
            return secrets.compare_digest(tok, session.token)

        def _coder(self, q: dict) -> str | None:
            c = (q.get("coder") or [""])[0]
            return c if CODER_RE.match(c) else None

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; img-src 'self' blob:; "
                                                              "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"})
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not self._authorized(q):
                return self._json({"error": "bad or missing token"}, 401)
            if u.path == "/api/session":
                coder = self._coder(q)
                if not coder:
                    return self._json({"error": "coder id must be 1-32 letters, digits, _ or -"}, 400)
                return self._json(session.session_payload(coder))
            m = re.fullmatch(r"/img/([0-9a-f]{16})", u.path)
            if m:
                rec = session.records.get(m.group(1))
                if rec is None or m.group(1) in session.hidden or not rec.image.local_path:
                    return self._json({"error": "not found"}, 404)
                p = Path(rec.image.local_path)
                if not p.exists():
                    return self._json({"error": "file missing"}, 404)
                return self._send(200, p.read_bytes(), rec.image.mime or "application/octet-stream")
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authorized(q):
                return self._json({"error": "bad or missing token"}, 401)
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 64_000)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json({"error": "bad json"}, 400)
            coder = body.get("coder")
            iid = body.get("item")
            if not (isinstance(coder, str) and CODER_RE.match(coder)) or iid not in session.records:
                return self._json({"error": "bad coder or item"}, 400)
            if u.path == "/api/motif":
                code, value = body.get("code"), body.get("value")
                if code not in session.codes or value not in (0, 1, "unsure", None):
                    return self._json({"error": "bad code or value"}, 400)
                session.append(coder, {"kind": "motif", "item": iid, "code": code, "value": value})
                return self._json({"ok": True})
            if u.path == "/api/affect":
                v = body.get("value")
                if v is not None:
                    ok = (isinstance(v, dict) and all(isinstance(v.get(k), int) and 1 <= v[k] <= 9 for k in ("valence", "arousal"))
                          and (v.get("dominance") is None or (isinstance(v["dominance"], int) and 1 <= v["dominance"] <= 9))
                          and v.get("primary_emotion") in session.cb.emotions
                          and v.get("secondary_emotion") in (*session.cb.emotions, None))
                    if not ok:
                        return self._json({"error": "bad affect value"}, 400)
                session.append(coder, {"kind": "affect", "item": iid, "value": v})
                return self._json({"ok": True})
            if u.path == "/api/flag":
                reason = body.get("reason")
                if reason not in FLAG_REASONS:
                    return self._json({"error": "bad reason"}, 400)
                session.flag(coder, iid, reason)
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

    return Handler


def serve(session: CodingSession, host: str = "0.0.0.0", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(session))
