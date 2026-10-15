"""Bluesky via the AT Protocol XRPC API - circulation-side corpus C.

Search requires an authenticated session: BSKY_HANDLE and BSKY_APP_PASSWORD
(an app password, never the account password).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterator

from ..engagement import snapshot
from ..models import Criterion, DetectionEvidence, ImageRecord, PostInfo, Rights, utcnow
from ..provenance import text_declares_ai
from .base import Collector, pseudonymize

PDS = "https://bsky.social/xrpc"


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(re.sub(r"Z$", "+00:00", s))


def _images(embed: dict | None) -> list[dict]:
    if not embed:
        return []
    t = embed.get("$type", "")
    if t.startswith("app.bsky.embed.images"):
        return embed.get("images", [])
    if t.startswith("app.bsky.embed.recordWithMedia"):
        return _images(embed.get("media"))
    return []


class BlueskyCollector(Collector):
    name = "bluesky"
    platform = "bluesky"

    def _auth(self) -> dict:
        handle, pw = os.environ.get("BSKY_HANDLE"), os.environ.get("BSKY_APP_PASSWORD")
        if not (handle and pw):
            raise RuntimeError("BSKY_HANDLE / BSKY_APP_PASSWORD not set")
        r = self.session.post(f"{PDS}/com.atproto.server.createSession", json={"identifier": handle, "password": pw})
        r.raise_for_status()
        return {"Authorization": f"Bearer {r.json()['accessJwt']}"}

    def collect(self) -> Iterator[ImageRecord]:
        p = self.frame.params
        headers = self._auth()
        for q in p.get("queries", []):
            cursor, n = None, 0
            while n < self.frame.limit:
                params = {"q": q, "limit": 100, "sort": p.get("sort", "latest")}
                for k in ("since", "until", "lang"):
                    if p.get(k):
                        params[k] = p[k]
                if cursor:
                    params["cursor"] = cursor
                r = self.session.get(f"{PDS}/app.bsky.feed.searchPosts", headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
                posts = data.get("posts", [])
                followers = self._followers(headers, {pp["author"]["did"] for pp in posts})
                for post in posts:
                    for rec in self.parse_post(post, query=q, followers=followers.get(post["author"]["did"])):
                        yield rec
                        n += 1
                cursor = data.get("cursor")
                if not cursor or not posts:
                    break

    def _followers(self, headers: dict, dids: set[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        dids_l = sorted(dids)
        for i in range(0, len(dids_l), 25):
            r = self.session.get(f"{PDS}/app.bsky.actor.getProfiles", headers=headers,
                                 params=[("actors", d) for d in dids_l[i:i + 25]])
            if r.ok:
                for prof in r.json().get("profiles", []):
                    out[prof["did"]] = prof.get("followersCount")
        return out

    def parse_post(self, post: dict, query: str | None = None, followers: int | None = None) -> list[ImageRecord]:
        imgs = _images(post.get("embed"))
        if not imgs:
            return []
        record = post.get("record", {})
        text = record.get("text", "")
        tags = sorted({f["tag"].lower() for facet in record.get("facets", []) for f in facet.get("features", [])
                       if f.get("$type") == "app.bsky.richtext.facet#tag"} | set(record.get("tags", [])))
        created = _parse_dt(record.get("createdAt") or post.get("indexedAt"))
        rkey = post["uri"].rsplit("/", 1)[-1]
        handle = post["author"].get("handle")
        labels = [lb.get("val") for lb in post.get("labels", []) if lb.get("val")]
        raw = {k: post.get(k) for k in ("likeCount", "repostCount", "replyCount", "quoteCount")}
        raw["followersCount"] = followers

        base_ev = []
        hit = text_declares_ai(text + " " + " ".join("#" + t for t in tags))
        if hit:
            base_ev.append(DetectionEvidence(criterion=Criterion.DECLARED, method="text:post", value=hit))

        out = []
        for i, im in enumerate(imgs):
            ev = list(base_ev)
            alt_hit = text_declares_ai(im.get("alt"))
            if alt_hit:
                ev.append(DetectionEvidence(criterion=Criterion.DECLARED, method="text:alt", value=alt_hit))
            out.append(ImageRecord(
                record_id=f"bluesky:{post['uri']}:{i}",
                image_index=i,
                source=self.source_kwargs(post["uri"], f"https://bsky.app/profile/{handle}/post/{rkey}", query),
                post=PostInfo(created_at=created, text=text, alt_text=im.get("alt"), hashtags=tags,
                              language=(record.get("langs") or [None])[0],
                              author_pseudonym=pseudonymize(post["author"]["did"]), platform_labels=labels),
                image={"media_url": im.get("fullsize")},
                rights=Rights(terms_basis="Bluesky Terms of Service / AT Protocol public data",
                              may_cache_bytes=True, may_redistribute_bytes=False),
                ai_evidence=ev,
                engagement=[snapshot("bluesky", raw, created, utcnow())],
            ))
        return out
