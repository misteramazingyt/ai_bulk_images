"""Wikimedia Commons - circulation-side corpus C (curated AI category tree).

Uses the MediaWiki Action API (categorymembers + imageinfo/extmetadata) and the
Wikimedia REST pageviews API. Follows the Wikimedia User-Agent policy and
API etiquette (maxlag, serial requests).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterator
from urllib.parse import quote

from ..engagement import snapshot
from ..models import Criterion, DetectionEvidence, ImageRecord, PostInfo, Rights, utcnow
from .base import Collector

API = "https://commons.wikimedia.org/w/api.php"
PV = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/commons.wikimedia.org/all-access/user"
FREE = re.compile(r"^(cc0|cc[- ]by(-sa)?|public domain|pd)", re.I)


class WikimediaCollector(Collector):
    name = "wikimedia_commons"
    platform = "wikimedia_commons"

    def _api(self, **params) -> dict:
        params.update(format="json", formatversion=2, maxlag=5)
        r = self.session.get(API, params=params)
        r.raise_for_status()
        return r.json()

    def _walk(self, root: str, depth: int) -> Iterator[tuple[str, str]]:
        seen, queue = set(), [(root, 0)]
        while queue:
            cat, d = queue.pop(0)
            if cat in seen:
                continue
            seen.add(cat)
            cont: dict = {}
            while True:
                data = self._api(action="query", list="categorymembers", cmtitle=cat, cmtype="file|subcat",
                                 cmlimit=500, **cont)
                for m in data["query"]["categorymembers"]:
                    if m["ns"] == 14 and d < depth:
                        queue.append((m["title"], d + 1))
                    elif m["ns"] == 6:
                        yield m["title"], cat
                if "continue" not in data:
                    break
                cont = data["continue"]

    def collect(self) -> Iterator[ImageRecord]:
        p = self.frame.params
        titles: list[tuple[str, str]] = []
        for root in p.get("categories", ["Category:AI-generated images"]):
            for t in self._walk(root, int(p.get("depth", 1))):
                titles.append(t)
                if len(titles) >= self.frame.limit:
                    break
        for i in range(0, len(titles), 50):
            batch = dict(titles[i:i + 50])
            data = self._api(action="query", prop="imageinfo", titles="|".join(batch),
                             iiprop="url|size|mime|extmetadata|timestamp")
            for page in data["query"]["pages"]:
                rec = self.parse_page(page, batch.get(page["title"]), with_pageviews=p.get("pageviews", True))
                if rec:
                    yield rec

    def _pageviews(self, title: str, days: int = 30) -> int | None:
        end = datetime.now(timezone.utc).date() - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        r = self.session.get(f"{PV}/{quote(title.replace(' ', '_'), safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
        if not r.ok:
            return None
        return sum(it.get("views", 0) for it in r.json().get("items", []))

    def parse_page(self, page: dict, category: str | None, with_pageviews: bool = False) -> ImageRecord | None:
        info = (page.get("imageinfo") or [None])[0]
        if not info or not str(info.get("mime", "")).startswith("image/"):
            return None
        meta = {k: v.get("value") for k, v in (info.get("extmetadata") or {}).items()}
        lic = meta.get("LicenseShortName")
        created = datetime.fromisoformat(info["timestamp"].replace("Z", "+00:00")) if info.get("timestamp") else None
        engagement = []
        if with_pageviews:
            engagement.append(snapshot("wikimedia_commons", {"pageviews": self._pageviews(page["title"]),
                                                             "window_days": 30}, created, utcnow()))
        return ImageRecord(
            record_id=f"wikimedia_commons:{page['pageid']}:0",
            source=self.source_kwargs(str(page["pageid"]), info.get("descriptionurl"), category),
            post=PostInfo(created_at=created, text=re.sub(r"<[^>]+>", "", meta.get("ImageDescription") or "")[:2000] or None,
                          community=category),
            image={"media_url": info.get("url"), "width": info.get("width"), "height": info.get("height"),
                   "mime": info.get("mime")},
            rights=Rights(license=lic, terms_basis=f"Wikimedia Commons file license: {lic}",
                          may_cache_bytes=True, may_redistribute_bytes=bool(lic and FREE.match(lic))),
            ai_evidence=[DetectionEvidence(criterion=Criterion.PLATFORM, method="commons_category", value=category)],
            engagement=engagement,
        )
