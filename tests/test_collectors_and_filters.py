from datetime import timedelta

import pytest

from aibi import codebook as cbmod
from aibi.collectors import Frame
from aibi.collectors.bluesky import BlueskyCollector
from aibi.collectors.factcheck import FactCheckCollector
from aibi.collectors.reddit import RedditCollector
from aibi.collectors.wikimedia import WikimediaCollector
from aibi.engagement import snapshot
from aibi.filters import evaluate
from aibi.models import AffectCode, AIStatus, Corpus, MotifCode
from aibi.storage import Store

from .test_core import T0, rec


@pytest.fixture(autouse=True)
def salt(monkeypatch):
    monkeypatch.setenv("AIBI_SALT", "test-salt")


def frame(src):
    return Frame(frame_id="t", source=src, corpus=Corpus.CIRCULATION)


def test_reddit_gallery_and_venue_rule():
    post = {
        "id": "abc", "created_utc": 1788000000, "subreddit": "midjourney", "title": "Storm over the capitol",
        "permalink": "/r/midjourney/comments/abc/x/", "author": "someone", "score": 120, "num_comments": 8,
        "num_crossposts": 2, "subreddit_subscribers": 1_500_000, "upvote_ratio": 0.97, "is_gallery": True,
        "gallery_data": {"items": [{"media_id": "m1"}, {"media_id": "m2"}]},
        "media_metadata": {"m1": {"e": "Image", "s": {"u": "https://i.redd.it/a.png?x=1&amp;y=2"}},
                           "m2": {"e": "Image", "s": {"u": "https://i.redd.it/b.png"}}},
    }
    recs = RedditCollector(frame("reddit")).parse_post(post, "r/midjourney")
    assert [r.record_id for r in recs] == ["reddit:abc:0", "reddit:abc:1"]
    assert recs[0].image.media_url == "https://i.redd.it/a.png?x=1&y=2"
    assert recs[0].ai_status is AIStatus.PROBABLE
    assert recs[0].engagement[0].standard.total == 130
    assert recs[0].post.author_pseudonym.startswith("u_") and "someone" not in recs[0].post.author_pseudonym


def test_reddit_skips_non_image():
    assert RedditCollector(frame("reddit")).parse_post({"id": "x", "post_hint": "link", "url": "https://e.com"}) == []


def test_bluesky_parse():
    post = {
        "uri": "at://did:plc:abc/app.bsky.feed.post/3kxyz", "cid": "c",
        "author": {"did": "did:plc:abc", "handle": "alice.bsky.social"},
        "record": {"text": "Flooded town #aiart", "createdAt": "2026-09-02T10:00:00.000Z", "langs": ["en"],
                   "facets": [{"features": [{"$type": "app.bsky.richtext.facet#tag", "tag": "AIArt"}]}]},
        "embed": {"$type": "app.bsky.embed.images#view",
                  "images": [{"fullsize": "https://cdn.bsky.app/img/full.jpg", "alt": "generated with Midjourney"}]},
        "likeCount": 40, "repostCount": 5, "replyCount": 3, "quoteCount": 1,
    }
    [r] = BlueskyCollector(frame("bluesky")).parse_post(post, "#aiart", followers=900)
    assert r.source.post_url == "https://bsky.app/profile/alice.bsky.social/post/3kxyz"
    assert r.post.hashtags == ["aiart"]
    assert {e.method for e in r.ai_evidence} == {"text:post", "text:alt"}
    assert r.engagement[0].standard.amplifications == 6 and r.engagement[0].standard.audience == 900


def test_wikimedia_parse():
    page = {"pageid": 42, "title": "File:X.png", "imageinfo": [{
        "url": "https://upload.wikimedia.org/x.png", "descriptionurl": "https://commons.wikimedia.org/wiki/File:X.png",
        "mime": "image/png", "width": 512, "height": 512, "timestamp": "2025-01-01T00:00:00Z",
        "extmetadata": {"LicenseShortName": {"value": "CC0"}, "ImageDescription": {"value": "<b>A</b> poster"}}}]}
    r = WikimediaCollector(frame("wikimedia_commons")).parse_page(page, "Category:AI-generated images")
    assert r.rights.may_redistribute_bytes and r.post.text == "A poster" and r.ai_status is AIStatus.PROBABLE


def test_factcheck_only_ai_findings_become_leads():
    claim = {"text": "Photo shows explosion near the Pentagon", "claimDate": "2023-05-22T00:00:00Z", "claimReview": [
        {"publisher": {"name": "AFP", "site": "factcheck.afp.com"}, "url": "https://factcheck.afp.com/x",
         "textualRating": "False - AI-generated image", "languageCode": "en"},
        {"publisher": {"name": "Other"}, "url": "https://o.example/y", "textualRating": "Misleading"}]}
    [r] = FactCheckCollector(frame("factcheck_claimreview")).parse_claim(claim, "q")
    assert r.exclusion == "lead_unconfirmed" and r.ai_status is AIStatus.CONFIRMED


def _coded(rid, platform, present, arousal, total, cluster):
    r = rec(rid, platform=platform, post={"created_at": T0})
    r.add_evidence(_declared())
    r.motif_codes.append(MotifCode(code_id="VM06", present=present, coder_id="h1", coder_type="human", codebook_version="0.1.0"))
    r.motif_codes.append(MotifCode(code_id="VM07", present=True, coder_id="clip", coder_type="model", codebook_version="0.1.0"))  # unvalidated: ignored
    r.affect_codes.append(AffectCode(coder_id="h1", coder_type="human", valence=2, arousal=arousal, codebook_version="0.1.0"))
    r.engagement.append(snapshot(platform, {"score": total, "likeCount": total}, T0, T0 + timedelta(days=8)))
    r.cluster_id = cluster
    return r


def _declared():
    from aibi.models import Criterion, DetectionEvidence

    return DetectionEvidence(criterion=Criterion.DECLARED, method="test")


def test_filters_end_to_end(tmp_path):
    cb = cbmod.load()
    recs = [_coded(f"r{i}", "reddit", present=i == 9, arousal=8 if i == 9 else 3, total=i, cluster=f"c{i}") for i in range(10)]
    recs.append(_coded("b0", "bluesky", present=True, arousal=8, total=1, cluster="c9"))  # same image on 2nd platform
    store = Store(tmp_path)
    store.upsert(recs)
    rows = {r.record_id: r for r in evaluate(store.iter_records(), cb)}
    assert rows["r9"].motifs == "VM06"  # VM07 from unvalidated model ignored
    assert rows["r9"].passes_filter1 and rows["r9"].passes_filter2
    assert rows["b0"].cluster_n_platforms == 2 and rows["b0"].passes_filter2
    assert not rows["r0"].passes_filter1 and not rows["r0"].passes_filter2


def test_store_merge_appends(tmp_path):
    store = Store(tmp_path)
    a = rec("x")
    a.engagement.append(snapshot("reddit", {"score": 1}, T0, T0 + timedelta(days=1)))
    store.upsert([a])
    b = rec("x")
    b.engagement.append(snapshot("reddit", {"score": 5}, T0, T0 + timedelta(days=8)))
    store.upsert([b])
    [merged] = list(store.iter_records())
    assert [s.standard.reactions for s in merged.engagement] == [1, 5]
