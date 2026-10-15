import csv
import json
import threading
import urllib.error
import urllib.request

import pytest

from aibi import codebook as cbmod
from aibi.coders.human import import_sheet
from aibi.storage import Store
from aibi.ui.server import CodingSession, export_coder, item_id, select_codes, serve

from .test_core import rec, scene

CB = cbmod.load()


@pytest.fixture
def app(tmp_path):
    store = Store(tmp_path / "data")
    recs = []
    for i in range(4):
        r = rec(f"reddit:p{i}:0", post={"text": "caption must not leak"})
        r.image.local_path = store.cache_image(f"{i:064x}", scene(i), ".png")
        r.image.mime = "image/png"
        recs.append(r)
    store.upsert(recs)
    session = CodingSession(store, CB, select_codes(CB, "VM04,VM06"), sample=3, seed=1, token="tok")
    httpd = serve(session, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None, token="tok"):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"X-Token": token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    yield store, session, call
    httpd.shutdown()


def test_select_codes():
    assert len(select_codes(CB, "visual")) == 20
    assert "PT10" not in select_codes(CB, "image") and "PT22" in select_codes(CB, "image")
    with pytest.raises(ValueError):
        select_codes(CB, "VM99")


def test_auth_blinding_and_orders(app):
    store, session, call = app
    assert call("/api/session?coder=A", token="wrong")[0] == 401
    assert call("/favicon.ico", token="")[0] == 204
    st, body = call("/api/session?coder=A")
    payload = json.loads(body)
    assert st == 200 and len(payload["items"]) == 3 and [c["id"] for c in payload["codes"]] == ["VM04", "VM06"]
    raw = body.decode()
    assert "reddit" not in raw and "caption" not in raw  # blind: no platform/caption leakage
    other = json.loads(call("/api/session?coder=B")[1])["items"]
    assert sorted(other) == sorted(payload["items"])  # same sample for every coder
    st, img = call(f"/img/{payload['items'][0]}")
    assert st == 200 and img[:4] == b"\x89PNG"
    assert call("/img/0000000000000000")[0] == 404
    assert call("/api/session?coder=bad%20id")[0] == 400


def test_coding_undo_export_and_flag(app, tmp_path):
    store, session, call = app
    a, b, c = json.loads(call("/api/session?coder=A")[1])["items"]
    for body in ({"item": a, "code": "VM04", "value": 1}, {"item": b, "code": "VM04", "value": 0},
                 {"item": b, "code": "VM06", "value": "unsure"}, {"item": a, "code": "VM06", "value": 1},
                 {"item": a, "code": "VM06", "value": None}):  # last one is an undo
        assert call("/api/motif", {"coder": "A", **body})[0] == 200
    assert call("/api/motif", {"coder": "A", "item": a, "code": "VM07", "value": 1})[0] == 400  # not in session
    bad = {"valence": 10, "arousal": 5, "primary_emotion": "fear"}
    assert call("/api/affect", {"coder": "A", "item": a, "value": bad})[0] == 400
    good = {"valence": 2, "arousal": 8, "primary_emotion": "fear", "secondary_emotion": None, "dominance": None}
    assert call("/api/affect", {"coder": "A", "item": a, "value": good})[0] == 200
    done = json.loads(call("/api/session?coder=A")[1])
    assert set(done["done_motifs"]) == {f"{a}|VM04", f"{b}|VM04", f"{b}|VM06"} and done["done_affect"] == [a]

    assert call("/api/flag", {"coder": "A", "item": c, "reason": "possible_minor_sexualization"})[0] == 200
    assert call(f"/img/{c}")[0] == 404
    assert c not in json.loads(call("/api/session?coder=B")[1])["items"]
    flagged = next(r for r in store.iter_records() if item_id(r.record_id) == c)
    assert flagged.exclusion == "csam_suspected_removed" and flagged.image.local_path is None

    out = tmp_path / "A.csv"
    assert export_coder(store, CB, "A", out) == 2
    rows = {r["record_id"]: r for r in csv.DictReader(out.open())}
    ra = next(r for rid, r in rows.items() if item_id(rid) == a)
    rb = next(r for rid, r in rows.items() if item_id(rid) == b)
    assert ra["VM04"] == "1" and ra["VM06"] == "" and ra["valence"] == "2" and ra["primary_emotion"] == "fear"
    assert rb["VM04"] == "0" and rb["VM06"] == "" and rb["note"] == "unsure:VM06"
    coded = import_sheet(out, CB, "A")  # round-trips into the standard importer
    assert {m.code_id for motifs, _ in coded.values() for m in motifs} == {"VM04"}
