"""Google Fact Check Tools API (ClaimReview) - verified-misuse corpus V.

Yields *leads*: fact-checks whose rating or title says the image is
AI-generated. ClaimReview carries no image, so `media_url` is a candidate
og:image from the review page (robots.txt respected). That image is often the
fact-checker's annotated copy ("FALSE" stamp, crops), so a human must confirm
or replace it with the circulating original before the record enters
analysis. Until then, `exclusion` = "lead_unconfirmed".
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterator

from ..models import Criterion, DetectionEvidence, ImageRecord, PostInfo, Rights
from .base import Collector

API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
AI_FINDING = re.compile(r"(ai[- ]generated|generated (with|by|using) (ai|artificial intelligence)|"
                        r"artificial intelligence|deepfake|synthetic image|fabricated by ai|midjourney|dall-?e)", re.I)
OG_IMAGE = re.compile(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', re.I)


class FactCheckCollector(Collector):
    name = "factcheck_claimreview"
    platform = "factcheck_claimreview"

    def collect(self) -> Iterator[ImageRecord]:
        key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_FACTCHECK_API_KEY not set")
        p = self.frame.params
        for q in p.get("queries", ["AI-generated image"]):
            token, n = None, 0
            while n < self.frame.limit:
                params = {"query": q, "pageSize": 100, "key": key, **({"pageToken": token} if token else {})}
                if p.get("languageCode"):
                    params["languageCode"] = p["languageCode"]
                if p.get("maxAgeDays"):
                    params["maxAgeDays"] = p["maxAgeDays"]
                r = self.session.get(API, params=params)
                r.raise_for_status()
                data = r.json()
                for claim in data.get("claims", []):
                    for rec in self.parse_claim(claim, q, fetch_og=p.get("fetch_og_image", True)):
                        yield rec
                        n += 1
                token = data.get("nextPageToken")
                if not token:
                    break

    def _og_image(self, url: str) -> str | None:
        try:
            r = self.session.get(url, respect_robots=True)
        except PermissionError:
            return None
        m = OG_IMAGE.search(r.text[:200_000]) if r.ok else None
        return m.group(1) if m else None

    def parse_claim(self, claim: dict, query: str, fetch_og: bool = False) -> list[ImageRecord]:
        out = []
        for i, rv in enumerate(claim.get("claimReview", [])):
            finding = " ".join(filter(None, [rv.get("textualRating"), rv.get("title")]))
            hit = AI_FINDING.search(finding)
            if not hit or not rv.get("url"):
                continue
            created = claim.get("claimDate") or rv.get("reviewDate")
            out.append(ImageRecord(
                record_id=f"factcheck_claimreview:{rv['url']}:0",
                source=self.source_kwargs(rv["url"], rv["url"], query),
                post=PostInfo(created_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
                              text=claim.get("text"), language=rv.get("languageCode"),
                              community=(rv.get("publisher") or {}).get("site")),
                image={"media_url": self._og_image(rv["url"]) if fetch_og else None},
                rights=Rights(terms_basis="Google Fact Check Tools API ToS; images: publisher copyright (fair-use research)",
                              may_cache_bytes=False, may_redistribute_bytes=False),
                ai_evidence=[DetectionEvidence(criterion=Criterion.FACTCHECK,
                                               method=f"claimreview:{(rv.get('publisher') or {}).get('name', '?')}",
                                               value=f"{rv.get('textualRating')} | matched='{hit.group(0)}'")],
                exclusion="lead_unconfirmed",
            ))
        return out
