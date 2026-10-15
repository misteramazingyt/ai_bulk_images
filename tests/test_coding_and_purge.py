import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from PIL import Image

from aibi import codebook as cbmod
from aibi import purge as purgemod
from aibi.calibration import apply_thresholds, calibrate, write_thresholds
from aibi.coders.vlm import SCORE, ClaudeVLMCoder, GeminiVLMCoder, batch_custom_id, prepare_image, response_schema
from aibi.coders.vlm_runner import run_sync
from aibi.filters import consensus_affect
from aibi.models import AffectCode, MotifCode
from aibi.storage import Store

from .test_core import png, rec, scene

CB = cbmod.load()


def model_output(present=("VM06",), flag="none", conf="high"):
    return {
        "codes": {cid: {"present": cid in present, "confidence": conf, "evidence": "x"} for cid in CB.codes},
        "valence": 2, "arousal": 8, "dominance": 3, "primary_emotion": "fear", "secondary_emotion": "sadness",
        "safety_flag": flag,
    }


class FakeMessages:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def create(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if out == "refusal":
            return SimpleNamespace(stop_reason="refusal", model="claude-opus-5-5", content=[],
                                   stop_details=SimpleNamespace(category="general_harms"))
        return SimpleNamespace(stop_reason="end_turn", model=out.pop("_model", "claude-opus-5-5"),
                               content=[SimpleNamespace(type="text", text=json.dumps(out))])


def fake_client(outputs):
    m = FakeMessages(outputs)
    return SimpleNamespace(beta=SimpleNamespace(messages=m), messages=m), m


def test_schema_valid_and_output_conforms():
    s = response_schema(CB)
    jsonschema.Draft202012Validator.check_schema(s)
    jsonschema.validate(model_output(), s)
    with pytest.raises(jsonschema.ValidationError):
        bad = model_output()
        del bad["codes"]["VM01"]
        jsonschema.validate(bad, s)


def test_prepare_image_downscales():
    media, data = prepare_image(png(size=(4000, 2000)))
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    assert media == "image/jpeg" and max(img.size) == 1568


def test_batch_custom_id_pattern():
    cid = batch_custom_id("bluesky:at://did:plc:abc/app.bsky.feed.post/3k:0")
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid)


def test_vlm_request_shape_and_parse():
    client, msgs = fake_client([model_output()])
    coder = ClaudeVLMCoder(CB, client=client)
    res = coder.code(scene(1))
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["messages"][0]["content"][0]["type"] == "image"
    present = {c.code_id for c in res.motif_codes if c.present}
    assert present == {"VM06"} and len(res.motif_codes) == len(CB.codes)
    assert all(not c.validated for c in res.motif_codes)  # no thresholds yet
    assert res.affect_codes[0].arousal == 8


def test_fallback_model_gets_its_own_coder_id():
    client, _ = fake_client([{**model_output(), "_model": "claude-opus-4-8"}])
    coder = ClaudeVLMCoder(CB, client=client)
    res = coder.code(scene(1))
    assert res.motif_codes[0].coder_id.startswith("vlm:claude-opus-4-8:")
    assert res.motif_codes[0].coder_id != coder.coder_id()


def _stored(tmp_path, n=1):
    store = Store(tmp_path / "data")
    recs = []
    for i in range(n):
        r = rec(f"r{i}")
        raw = scene(i)
        r.image.local_path = store.cache_image(f"{i:064x}", raw, ".png")
        recs.append(r)
    store.upsert(recs)
    return store


def test_run_sync_codes_skips_refusals_and_is_idempotent(tmp_path):
    store = _stored(tmp_path, 2)
    client, msgs = fake_client([model_output(), "refusal"])
    coder = ClaudeVLMCoder(CB, client=client)
    stats = run_sync(store, coder)
    assert stats == {"coded": 1, "refused": 1, "errors": 0}
    client.beta.messages.outputs = [model_output()]
    assert run_sync(store, coder)["coded"] == 1  # only the previously refused one is retried
    assert len(msgs.calls) == 3


def test_safety_flag_deletes_bytes_and_excludes(tmp_path):
    store = _stored(tmp_path)
    path = next(store.iter_records()).image.local_path
    client, _ = fake_client([model_output(flag="possible_minor_sexualization")])
    run_sync(store, ClaudeVLMCoder(CB, client=client))
    r = next(store.iter_records())
    assert not Path(path).exists() and r.image.local_path is None
    assert r.exclusion == "csam_suspected_removed"


def test_calibration_end_to_end(tmp_path):
    coder_id = "vlm:test"
    recs = []
    for i in range(80):
        r = rec(f"r{i}")
        truth = i % 2 == 0
        r.motif_codes.append(MotifCode(code_id="VM06", present=truth, coder_id="h", coder_type="human", codebook_version="0.1.0"))
        # model: right 90% of the time, with ordinal scores
        guess = truth if i % 10 else not truth
        r.motif_codes.append(MotifCode(code_id="VM06", present=guess, score=SCORE[(guess, "high")], coder_id=coder_id,
                                       coder_type="model", codebook_version="0.1.0"))
        r.affect_codes.append(AffectCode(coder_id="h", coder_type="human", valence=2 + i % 5, arousal=3 + i % 5, codebook_version="0.1.0"))
        r.affect_codes.append(AffectCode(coder_id=coder_id, coder_type="model", valence=2 + i % 5, arousal=3 + i % 5, codebook_version="0.1.0"))
        recs.append(r)
    report = calibrate(recs, CB, coder_id, seed=1)
    vm06 = report["codes"]["VM06"]
    assert vm06["validated"] and vm06["test"]["f1"] >= 0.7 and vm06["n_test"] + vm06["n_calibration"] == 80
    assert report["affect"]["validated"]
    path = tmp_path / "thresholds.json"
    write_thresholds(report, path)
    th = json.loads(path.read_text())
    assert apply_thresholds(recs, th) == 80
    assert all(c.validated for r in recs for c in r.motif_codes if c.coder_type == "model")
    stripped = rec("solo")
    stripped.affect_codes = [a for a in recs[0].affect_codes if a.coder_type == "model"]
    assert consensus_affect(stripped) == (2, 3)  # validated model affect used when no humans


def test_calibration_refuses_with_too_few_positives():
    r = rec("a")
    r.motif_codes += [MotifCode(code_id="VM01", present=True, coder_id="h", coder_type="human", codebook_version="0"),
                      MotifCode(code_id="VM01", present=True, score=0.95, coder_id="m", coder_type="model", codebook_version="0")]
    rep = calibrate([r], CB, "m")
    assert rep["codes"]["VM01"]["validated"] is False


def test_purge_tombstones_and_keeps_shared_bytes(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    shared = store.cache_image("ab" * 32, scene(1), ".png")
    a = rec("reddit:gone:0", post={"text": "secret", "community": "midjourney"})
    a.source.native_id = "gone"
    a.image.local_path = shared
    b = rec("reddit:kept:0")
    b.source.native_id = "kept"
    b.image.local_path = shared  # same image cross-posted
    store.upsert([a, b])
    monkeypatch.setitem(purgemod.CHECKERS, "reddit", lambda ids, s: ({"gone"}, 0))
    rep = purgemod.purge(store, session=None)
    assert rep["reddit"]["records_tombstoned"] == 1
    recs = store.load("reddit")
    assert recs["reddit:gone:0"].exclusion.startswith("user_deleted@")
    assert recs["reddit:gone:0"].post.text is None and recs["reddit:gone:0"].image.local_path is None
    assert Path(shared).exists()  # still referenced by the live record
    rep2 = purgemod.purge(store, session=None)
    assert rep2["reddit"]["checked"] == 1  # tombstoned items are not re-checked


class FakeGemini:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate_content(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def gemini_resp(out=None, finish="STOP", block=None):
    return SimpleNamespace(
        model_version="gemini-3.1-pro-preview",
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        candidates=[] if block else [SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        text=json.dumps(out) if out else "",
    )


def test_gemini_coder_request_and_blocks():
    fake = FakeGemini([gemini_resp(model_output(present=("VM04",))), gemini_resp(finish="SAFETY"),
                       gemini_resp(block=SimpleNamespace(name="PROHIBITED_CONTENT"))])
    coder = GeminiVLMCoder(CB, client=SimpleNamespace(models=fake))
    res = coder.code(scene(1))
    cfg = fake.calls[0]["config"]
    assert cfg.response_json_schema == response_schema(CB) and cfg.response_mime_type == "application/json"
    assert cfg.thinking_config.thinking_level.name == "HIGH"
    assert {c.code_id for c in res.motif_codes if c.present} == {"VM04"}
    assert res.motif_codes[0].coder_id.startswith("vlm:gemini-3.1-pro-preview:")
    assert coder.code(scene(1)).refused == "finish:SAFETY"
    assert coder.code(scene(1)).refused == "blocked:PROHIBITED_CONTENT"


def test_provider_settings_change_coder_id():
    g = GeminiVLMCoder(CB, client=object())
    c = ClaudeVLMCoder(CB, client=object())
    assert g.prompt_hash != c.prompt_hash  # provider config is part of the fingerprint
    assert GeminiVLMCoder(CB, effort="low", client=object()).coder_id() != g.coder_id()


def test_batch_rejected_for_gemini(tmp_path):
    from aibi.coders.vlm_runner import submit_batch

    with pytest.raises(ValueError):
        submit_batch(_stored(tmp_path), GeminiVLMCoder(CB, client=object()))
