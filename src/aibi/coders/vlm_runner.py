"""Run the VLM coder over the store, synchronously or via the Batches API."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

from ..models import ImageRecord, utcnow
from ..safety import apply_safety_flag
from ..storage import Store
from .vlm import VLMCoder, VLMResult, batch_custom_id

log = logging.getLogger("aibi.vlm")


def pending(store: Store, coder: VLMCoder, limit: int | None, record_ids: set[str] | None = None) -> list[ImageRecord]:
    """Records with cached bytes, not excluded, not yet coded by this prompt version."""
    out = []
    for rec in store.iter_records():
        if rec.exclusion or not rec.image.local_path:
            continue
        if record_ids is not None and rec.record_id not in record_ids:
            continue
        if any(c.coder_id.endswith(f":p{coder.prompt_hash}") for c in rec.motif_codes):
            continue
        out.append(rec)
        if limit and len(out) >= limit:
            break
    return out


def apply_result(rec: ImageRecord, res: VLMResult) -> None:
    rec.motif_codes.extend(res.motif_codes)
    rec.affect_codes.extend(res.affect_codes)
    apply_safety_flag(rec, res.safety_flag, source=res.motif_codes[0].coder_id if res.motif_codes else "vlm")


def _save(store: Store, recs: Iterable[ImageRecord]) -> None:
    by_plat: dict[str, dict[str, ImageRecord]] = {}
    for r in recs:
        by_plat.setdefault(r.source.platform, {})[r.record_id] = r
    for plat, updated in by_plat.items():
        allrecs = store.load(plat)
        allrecs.update(updated)
        store.write(plat, allrecs.values())


def run_sync(store: Store, coder: VLMCoder, limit: int | None = None, record_ids: set[str] | None = None) -> dict:
    import anthropic

    stats = {"coded": 0, "refused": 0, "errors": 0}
    done: list[ImageRecord] = []
    for rec in pending(store, coder, limit, record_ids):
        try:
            res = coder.code(Path(rec.image.local_path).read_bytes())
        except (anthropic.BadRequestError, anthropic.NotFoundError) as e:
            log.error("%s: non-retryable API error: %s", rec.record_id, e)
            stats["errors"] += 1
            continue
        except anthropic.APIError as e:  # SDK already retried 429/5xx/connection errors
            log.error("%s: API error after retries: %s", rec.record_id, e)
            stats["errors"] += 1
            continue
        if res.refused:
            log.warning("%s: refused (%s)", rec.record_id, res.refused)
            stats["refused"] += 1
            continue
        apply_result(rec, res)
        done.append(rec)
        stats["coded"] += 1
        if len(done) % 25 == 0:
            _save(store, done)
    _save(store, done)
    return stats


def submit_batch(store: Store, coder: VLMCoder, limit: int | None = None) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    recs = pending(store, coder, limit)
    if not recs:
        return {"submitted": 0}
    mapping = {batch_custom_id(r.record_id): r.record_id for r in recs}
    requests = [Request(custom_id=batch_custom_id(r.record_id),
                        params=MessageCreateParamsNonStreaming(**coder.params(Path(r.image.local_path).read_bytes())))
                for r in recs]
    batch = coder.client.messages.batches.create(requests=requests)
    state = {"batch_id": batch.id, "created_at": utcnow().isoformat(), "model": coder.model, "effort": coder.effort,
             "prompt_hash": coder.prompt_hash, "codebook_version": coder.cb.version, "custom_ids": mapping}
    path = store.root / "vlm_batches" / f"{batch.id}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return {"submitted": len(requests), "batch_id": batch.id, "state_file": str(path)}


def collect_batch(store: Store, coder: VLMCoder, batch_id: str, wait: bool = False, poll_seconds: int = 60) -> dict:
    state = json.loads((store.root / "vlm_batches" / f"{batch_id}.json").read_text())
    if state["prompt_hash"] != coder.prompt_hash:
        raise RuntimeError("codebook/prompt changed since this batch was submitted; collect with the matching version")
    while True:
        batch = coder.client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if not wait:
            return {"status": batch.processing_status, "counts": batch.request_counts.to_dict()}
        time.sleep(poll_seconds)

    records = {r.record_id: r for r in store.iter_records()}
    stats = {"coded": 0, "refused": 0, "errored": 0, "expired_or_canceled": 0}
    retry: list[str] = []
    done = []
    for result in coder.client.messages.batches.results(batch_id):
        rid = state["custom_ids"].get(result.custom_id)
        rec = records.get(rid)
        if rec is None:
            continue
        kind = result.result.type
        if kind == "succeeded":
            res = coder.parse_message(result.result.message)
            if res.refused:
                stats["refused"] += 1
                retry.append(rid)  # re-run synchronously, where refusal fallbacks are available
                continue
            apply_result(rec, res)
            done.append(rec)
            stats["coded"] += 1
        elif kind == "errored":
            stats["errored"] += 1
            retry.append(rid)
        else:
            stats["expired_or_canceled"] += 1
            retry.append(rid)
    _save(store, done)
    stats["retry_record_ids"] = retry
    return stats
