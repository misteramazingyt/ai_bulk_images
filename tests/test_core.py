import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from aibi import codebook as cbmod
from aibi.engagement import normalize, percentile_rank, snapshot, standardize
from aibi.hashing import cluster, hamming, perceptual_hashes, sha256
from aibi.models import AIStatus, Corpus, Criterion, DetectionEvidence, ImageRecord, Rights, SourceInfo
from aibi.provenance import inspect_bytes, text_declares_ai
from aibi.reliability import auroc, calibrate_threshold, krippendorff_alpha

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def png(color=(200, 30, 30), size=(64, 64), text=None, pattern=False) -> bytes:
    img = Image.new("RGB", size, color)
    if pattern:
        for x in range(size[0]):
            for y in range(size[1]):
                if (x // 8 + y // 8) % 2:
                    img.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def rec(rid, platform="reddit", **kw) -> ImageRecord:
    return ImageRecord(
        record_id=rid,
        source=SourceInfo(platform=platform, corpus=Corpus.CIRCULATION, sampling_frame="t", collector="t",
                          collector_version="0", native_id=rid),
        rights=Rights(terms_basis="test"),
        **kw,
    )


# --- codebook -------------------------------------------------------------

def test_codebook_loads_and_is_complete():
    cb = cbmod.load()
    assert len(cb.visual_motifs) == 20
    assert len(cb.techniques) == 22  # SemEval-2021 Task 6 taxonomy
    assert cb.lexicon_matches("a crying child in a flood near a destroyed city") == ["VM06", "VM07"]


def test_codebook_doc_in_sync():
    doc = (ROOT / "docs" / "CODEBOOK.md").read_text()
    assert cbmod.render_markdown(cbmod.load()) in doc, "run `aibi codebook-doc`"


def test_schema_file_in_sync():
    on_disk = json.loads((ROOT / "schema" / "image_record.schema.json").read_text())
    assert on_disk == ImageRecord.model_json_schema(), "run `aibi schema`"


# --- AI-status derivation and provenance ----------------------------------

def test_ai_status_takes_strongest_evidence():
    r = rec("a")
    assert r.ai_status is AIStatus.UNVERIFIED
    r.add_evidence(DetectionEvidence(criterion=Criterion.DETECTOR, method="x"))
    assert r.ai_status is AIStatus.POSSIBLE
    r.add_evidence(DetectionEvidence(criterion=Criterion.DECLARED, method="venue"))
    assert r.ai_status is AIStatus.PROBABLE
    r.add_evidence(DetectionEvidence(criterion=Criterion.META, method="png"))
    assert r.ai_status is AIStatus.CONFIRMED
    r.add_evidence(DetectionEvidence(criterion=Criterion.META, method="png"))
    assert len(r.ai_evidence) == 3  # deduplicated


def test_png_generator_metadata_detected():
    ev = inspect_bytes(png(text={"parameters": "a flag, Steps: 20, Sampler: Euler a, Seed: 1"}))
    assert [e.method for e in ev] == ["png_text_chunk:parameters"]
    assert ev[0].criterion is Criterion.META
    assert inspect_bytes(png()) == []


def test_xmp_digital_source_type_detected():
    xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description Iptc4xmpExt:DigitalSourceType='
           '"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>')
    ev = inspect_bytes(png(text={"XML:com.adobe.xmp": xmp}))
    assert any(e.value == "trainedAlgorithmicMedia" for e in ev)


@pytest.mark.parametrize("text,hit", [
    ("Made with Midjourney v7", True), ("#aiart #fantasy", True), ("AI-generated image of a storm", True),
    ("Look at this flux of people", False), ("my holiday photo", False), ("stable diffusion test", True),
])
def test_text_declaration(text, hit):
    assert bool(text_declares_ai(text)) is hit


# --- hashing --------------------------------------------------------------

def scene(seed: int) -> bytes:
    """A non-degenerate test image: gradient background plus random ellipses."""
    import random

    from PIL import ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (128, 128))
    for y in range(128):
        for x in range(128):
            img.putpixel((x, y), (x * 2, y * 2, (x + y) % 256))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = rng.randrange(100), rng.randrange(100)
        d.ellipse([x0, y0, x0 + rng.randrange(10, 40), y0 + rng.randrange(10, 40)],
                  fill=tuple(rng.randrange(256) for _ in range(3)))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_near_duplicate_clustering():
    a = scene(1)
    b_img = Image.open(io.BytesIO(a)).resize((100, 100))
    buf = io.BytesIO()
    b_img.convert("RGB").save(buf, "JPEG", quality=70)
    b = buf.getvalue()
    c = Image.open(io.BytesIO(scene(2))).rotate(90)
    buf = io.BytesIO()
    c.save(buf, "PNG")
    c = buf.getvalue()
    pa, pb, pc = (perceptual_hashes(x)[0] for x in (a, b, c))
    assert hamming(pa, pb) <= 8
    ids = cluster([("a", sha256(a), pa), ("b", sha256(b), pb), ("c", sha256(c), pc), ("a2", sha256(a), None)])
    assert ids["a"] == ids["b"] == ids["a2"]
    assert ids["c"] != ids["a"]


def test_cluster_banding_is_exact():
    import random

    rng = random.Random(1)
    base = rng.getrandbits(64)
    items = [("base", None, f"{base:016x}")]
    for d in range(0, 9):
        h = base
        for bit in rng.sample(range(64), d):
            h ^= 1 << bit
        items.append((f"d{d}", None, f"{h:016x}"))
    far = base ^ int("F" * 4, 16) ^ (0xFFFF << 32)  # 32 bits away
    items.append(("far", None, f"{far:016x}"))
    ids = cluster(items, max_distance=8)
    assert all(ids[f"d{d}"] == ids["base"] for d in range(9))
    assert ids["far"] != ids["base"]


# --- engagement -----------------------------------------------------------

def test_standardize_mappings():
    s = standardize("bluesky", {"likeCount": 10, "repostCount": 2, "quoteCount": 1, "replyCount": 4, "followersCount": 100})
    assert (s.reactions, s.amplifications, s.replies, s.audience, s.total) == (10, 3, 4, 100, 17)
    r = standardize("reddit", {"score": 50, "num_comments": 5, "num_crossposts": 1, "subreddit_subscribers": 1000})
    assert r.total == 56


def test_percentile_ties():
    assert percentile_rank([1, 1, 3]) == pytest.approx([33.333, 33.333, 83.333], rel=1e-3)


def test_normalize_uses_t7d_snapshot_and_fallback_strata():
    recs = []
    for i in range(10):
        r = rec(f"r{i}", post={"created_at": T0, "community": "sub"})
        r.engagement.append(snapshot("reddit", {"score": 999}, T0, T0 + timedelta(hours=2)))  # too young
        r.engagement.append(snapshot("reddit", {"score": i, "num_comments": 0}, T0, T0 + timedelta(days=8)))
        recs.append(r)
    out = normalize(recs, min_stratum_size=30)
    assert out["r9"].total == 9
    assert out["r9"].stratum == "reddit|*|2026-09"  # fell back: stratum n < 30
    assert out["r9"].percentile > out["r0"].percentile


# --- reliability ----------------------------------------------------------

def test_krippendorff_reference_values():
    # Krippendorff (2011) "Computing Krippendorff's alpha-reliability", nominal example: alpha = 0.743
    data = [
        [1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
        [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1], [None, None, 3, None],
    ]
    assert krippendorff_alpha(data, "nominal") == pytest.approx(0.743, abs=1e-3)
    assert krippendorff_alpha(data, "interval") == pytest.approx(0.849, abs=1e-3)
    assert krippendorff_alpha([[1, 1], [0, 0], [1, 1]], "nominal") == 1.0


def test_calibration():
    y = [True, True, True, False, False, False]
    s = [0.9, 0.8, 0.4, 0.5, 0.2, 0.1]
    m = calibrate_threshold(y, s)
    assert m.threshold == 0.4 and m.f1 == pytest.approx(0.857, abs=1e-3)
    assert auroc(y, s) == pytest.approx(8 / 9)
