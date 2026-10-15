"""Deletion compliance (ETHICS_AND_LEGAL §2): re-check items at source and
tombstone those the author or platform has removed.

Run before every analysis snapshot and before any data release. A record is
tombstoned only on positive evidence of deletion/removal. Transient errors
leave it untouched, and those errors are counted in the report.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from .collectors import Frame
from .collectors.bluesky import PDS, BlueskyCollector
from .collectors.reddit import API as REDDIT_API
from .collectors.reddit import RedditCollector
from .collectors.wikimedia import WikimediaCollector
from .http import PoliteSession
from .models import Corpus, ImageRecord
from .safety import tombstone
from .storage import Store

log = logging.getLogger("aibi.purge")
REDDIT_REMOVED = {"deleted", "author", "moderator", "reddit", "automod_filtered", "anti_evil_ops",
                  "copyright_takedown", "content_takedown", "community_ops", "legal_operations"}


def _frame(source: str) -> Frame:
    return Frame(frame_id="purge", source=source, corpus=Corpus.CIRCULATION)


def reddit_deleted(native_ids: list[str], session: PoliteSession) -> tuple[set[str], int]:
    c = RedditCollector(_frame("reddit"), session=session)
    headers = {"Authorization": f"bearer {c._token()}"}
    gone, errors = set(), 0
    for i in range(0, len(native_ids), 100):
        chunk = native_ids[i:i + 100]
        r = session.get(f"{REDDIT_API}/api/info", headers=headers,
                        params={"id": ",".join("t3_" + n for n in chunk), "raw_json": 1})
        if not r.ok:
            errors += len(chunk)
            continue
        seen = set()
        for child in r.json()["data"]["children"]:
            d = child["data"]
            seen.add(d["id"])
            if d.get("author") == "[deleted]" or d.get("removed_by_category") in REDDIT_REMOVED:
                gone.add(d["id"])
        gone |= set(chunk) - seen  # not returned at all
    return gone, errors


def bluesky_deleted(uris: list[str], session: PoliteSession) -> tuple[set[str], int]:
    headers = BlueskyCollector(_frame("bluesky"), session=session)._auth()
    gone, errors = set(), 0
    for i in range(0, len(uris), 25):
        chunk = uris[i:i + 25]
        r = session.get(f"{PDS}/app.bsky.feed.getPosts", headers=headers, params=[("uris", u) for u in chunk])
        if not r.ok:
            errors += len(chunk)
            continue
        present = {p["uri"] for p in r.json().get("posts", [])}
        gone |= set(chunk) - present
    return gone, errors


def commons_deleted(pageids: list[str], session: PoliteSession) -> tuple[set[str], int]:
    c = WikimediaCollector(_frame("wikimedia_commons"), session=session)
    gone, errors = set(), 0
    for i in range(0, len(pageids), 50):
        chunk = pageids[i:i + 50]
        try:
            data = c._api(action="query", pageids="|".join(chunk))
        except Exception:
            errors += len(chunk)
            continue
        for page in data["query"]["pages"]:
            if page.get("missing"):
                gone.add(str(page["pageid"]))
    return gone, errors


CHECKERS: dict[str, Callable[[list[str], PoliteSession], tuple[set[str], int]]] = {
    "reddit": reddit_deleted,
    "bluesky": bluesky_deleted,
    "wikimedia_commons": commons_deleted,
}


def purge(store: Store, session: PoliteSession, platforms: Iterable[str] | None = None, dry_run: bool = False) -> dict:
    report = {}
    all_recs = {p: store.load(p) for p in store.platforms()}
    for plat in platforms or list(all_recs):
        if plat not in CHECKERS or plat not in all_recs:
            continue
        recs = all_recs[plat]
        live = [r for r in recs.values() if not (r.exclusion or "").startswith("user_deleted")]
        ids = sorted({r.source.native_id for r in live})
        gone, errors = CHECKERS[plat](ids, session) if ids else (set(), 0)
        hit: list[ImageRecord] = [r for r in live if r.source.native_id in gone]
        if not dry_run:
            hit_ids = {r.record_id for r in hit}
            still_used = {r.image.local_path for p in all_recs.values() for r in p.values()
                          if r.record_id not in hit_ids and r.image.local_path
                          and not (r.exclusion or "").startswith("user_deleted")}
            for r in hit:
                tombstone(r, "user_deleted", delete_file=r.image.local_path not in still_used)
            store.write(plat, recs.values())
        report[plat] = {"checked": len(ids), "deleted_at_source": len(gone), "records_tombstoned": len(hit),
                        "check_errors": errors, "dry_run": dry_run}
        log.info("purge %s: %s", plat, report[plat])
    return report
