"""Reddit Data API (OAuth, app-only) - circulation-side corpus C.

Requires a registered app: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and a
descriptive REDDIT_USER_AGENT. Collection must follow the Reddit Data API
Terms, including deleting content that users remove (see `aibi purge`).
"""

from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from typing import Iterator

from ..engagement import snapshot
from ..models import Criterion, DetectionEvidence, ImageRecord, PostInfo, Rights, utcnow
from ..provenance import text_declares_ai
from .base import Collector, pseudonymize

API = "https://oauth.reddit.com"

# Communities whose rules require AI-generated content (criterion DECLARED,
# method venue_rule). Verify each community's current rules before adding.
AI_ONLY_SUBREDDITS = {"midjourney", "stablediffusion", "dalle2", "aiart", "weirddalle", "midjourneyfantasy", "aiartwork"}


class RedditCollector(Collector):
    name = "reddit"
    platform = "reddit"

    def _token(self) -> str:
        cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
        if not (cid and secret):
            raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
        ua = os.environ.get("REDDIT_USER_AGENT")
        if ua:
            self.session.s.headers["User-Agent"] = ua
        r = self.session.post("https://www.reddit.com/api/v1/access_token", auth=(cid, secret),
                              data={"grant_type": "client_credentials"})
        r.raise_for_status()
        return r.json()["access_token"]

    def collect(self) -> Iterator[ImageRecord]:
        p = self.frame.params
        headers = {"Authorization": f"bearer {self._token()}"}
        sort, per_sub = p.get("sort", "new"), self.frame.limit
        for sub in p.get("subreddits", []):
            after, n = None, 0
            while n < per_sub:
                params = {"limit": 100, "raw_json": 1, **({"after": after} if after else {})}
                if p.get("q"):
                    url = f"{API}/r/{sub}/search"
                    params.update(q=p["q"], restrict_sr=1, sort=sort, t=p.get("t", "all"))
                else:
                    url = f"{API}/r/{sub}/{sort}"
                r = self.session.get(url, headers=headers, params=params)
                r.raise_for_status()
                data = r.json()["data"]
                for child in data["children"]:
                    for rec in self.parse_post(child["data"], query=f"r/{sub}"):
                        yield rec
                        n += 1
                after = data.get("after")
                if not after:
                    break

    def parse_post(self, d: dict, query: str | None = None) -> list[ImageRecord]:
        urls: list[str] = []
        if d.get("is_gallery") and d.get("media_metadata"):
            for item in (d.get("gallery_data") or {}).get("items", []):
                meta = d["media_metadata"].get(item["media_id"], {})
                if meta.get("e") == "Image" and "s" in meta:
                    urls.append(html.unescape(meta["s"].get("u", "")))
        elif d.get("post_hint") == "image" and d.get("url"):
            urls.append(d["url"])
        if not urls:
            return []

        created = datetime.fromtimestamp(d["created_utc"], tz=timezone.utc)
        now = utcnow()
        sub = d.get("subreddit", "")
        raw_eng = {k: d.get(k) for k in ("score", "upvote_ratio", "num_comments", "num_crossposts", "subreddit_subscribers")}
        evidence = []
        if sub.lower() in AI_ONLY_SUBREDDITS:
            evidence.append(DetectionEvidence(criterion=Criterion.DECLARED, method=f"venue_rule:r/{sub}"))
        flair = d.get("link_flair_text") or ""
        for field, text in (("title", d.get("title")), ("flair", flair)):
            hit = text_declares_ai(text)
            if hit:
                evidence.append(DetectionEvidence(criterion=Criterion.DECLARED, method=f"text:{field}", value=hit))

        out = []
        for i, u in enumerate(urls):
            out.append(ImageRecord(
                record_id=f"reddit:{d['id']}:{i}",
                image_index=i,
                source=self.source_kwargs(d["id"], "https://www.reddit.com" + d.get("permalink", ""), query),
                post=PostInfo(created_at=created, text=d.get("title"), community=sub,
                              author_pseudonym=pseudonymize(d.get("author_fullname") or d.get("author")),
                              platform_labels=[flair] if flair else [], nsfw=bool(d.get("over_18"))),
                image={"media_url": u},
                rights=Rights(terms_basis="Reddit Data API Terms (non-commercial research)",
                              may_cache_bytes=True, may_redistribute_bytes=False),
                ai_evidence=list(evidence),
                engagement=[snapshot("reddit", raw_eng, created, now)],
            ))
        return out
