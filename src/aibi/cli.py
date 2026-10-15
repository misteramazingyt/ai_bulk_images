"""Command-line entry point: `aibi <command> --help`."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml

from . import codebook as cbmod
from .models import ImageRecord
from .storage import Store


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def cmd_collect(a):
    from .pipeline import collect

    m = collect(_cfg(a.config), Store(a.data), frame_ids=a.frame or None, fetch=not a.no_fetch)
    print(json.dumps({"run_id": m["run_id"], "frames": [{f["frame_id"]: f["counts"]} for f in m["frames"]],
                      "errors": m["errors"]}, indent=2, default=str))


def cmd_cluster(a):
    from .pipeline import run_clustering

    cb = cbmod.load(a.codebook)
    print(json.dumps(run_clustering(Store(a.data), a.max_distance or cb.filters.get("phash_max_distance", 8))))


def cmd_code_clip(a):
    from .coders.clip_zeroshot import ClipZeroShotCoder

    cb, store = cbmod.load(a.codebook), Store(a.data)
    coder = ClipZeroShotCoder(cb, a.model, a.pretrained, a.thresholds)
    for plat in store.platforms():
        recs = store.load(plat)
        for r in recs.values():
            if r.image.local_path and not any(c.coder_id == coder.coder_id for c in r.motif_codes):
                r.motif_codes.extend(coder.code(Path(r.image.local_path).read_bytes()))
        store.write(plat, recs.values())


def cmd_sheet(a):
    from .coders.human import export_sheet

    n = export_sheet(Store(a.data).iter_records(), cbmod.load(a.codebook), a.out, a.sample, a.seed)
    print(f"wrote {n} rows to {a.out}")


def cmd_import_codes(a):
    from .coders.human import import_sheet

    cb, store = cbmod.load(a.codebook), Store(a.data)
    coded = import_sheet(a.sheet, cb, a.coder_id)
    for plat in store.platforms():
        recs = store.load(plat)
        for rid, (motifs, affect) in coded.items():
            if rid in recs:
                recs[rid].motif_codes.extend(motifs)
                recs[rid].affect_codes.extend(affect)
        store.write(plat, recs.values())
    print(f"imported codes for {len(coded)} records")


def cmd_reliability(a):
    """Compute alpha per variable from 2+ coded sheets with the same record_ids."""
    from .reliability import interpret_alpha, krippendorff_alpha

    cb = cbmod.load(a.codebook)
    sheets = []
    for p in a.sheets:
        with open(p, newline="") as f:
            sheets.append({r["record_id"]: r for r in csv.DictReader(f)})
    ids = sorted(set.intersection(*(set(s) for s in sheets)))
    report = {}
    variables = [(c, "nominal") for c in cb.codes] + [("valence", "interval"), ("arousal", "interval"),
                                                      ("primary_emotion", "nominal")]
    for var, level in variables:
        units = []
        for rid in ids:
            vals = [(s[rid].get(var) or "").strip() or None for s in sheets]
            if level == "interval":
                vals = [float(v) if v is not None else None for v in vals]
            units.append(vals)
        alpha = krippendorff_alpha(units, level)
        n_pairable = sum(sum(v is not None for v in u) >= 2 for u in units)
        if n_pairable:
            report[var] = {"alpha": round(alpha, 3), "level": level, "n_units": n_pairable,
                           "verdict": interpret_alpha(alpha) if alpha == alpha else "undefined"}
    print(json.dumps(report, indent=2))


def cmd_filter(a):
    from .filters import evaluate, rows_to_dicts

    rows = rows_to_dicts(evaluate(Store(a.data).iter_records(), cbmod.load(a.codebook)))
    if a.only_passing:
        rows = [r for r in rows if r["passes_filter1"] and r["passes_filter2"]]
    out = open(a.out, "w", newline="") if a.out else sys.stdout
    if rows:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows", file=sys.stderr)


def cmd_schema(a):
    Path(a.out).write_text(json.dumps(ImageRecord.model_json_schema(), indent=2) + "\n")


def cmd_codebook_doc(a):
    path = Path(a.out)
    text = path.read_text()
    marker = "<!-- BEGIN GENERATED -->"
    head = text.split(marker)[0] if marker in text else text + "\n"
    path.write_text(head + marker + "\n\n" + cbmod.render_markdown(cbmod.load(a.codebook)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="aibi", description=__doc__)
    p.add_argument("--data", default="data", help="data directory (default: data/)")
    p.add_argument("--codebook", default=str(cbmod.DEFAULT_PATH))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("collect", help="run sampling frames from config")
    s.add_argument("--config", default="config.yaml")
    s.add_argument("--frame", action="append", help="frame_id to run (repeatable); default all")
    s.add_argument("--no-fetch", action="store_true", help="collect metadata only")
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("cluster", help="assign near-duplicate clusters across all platforms")
    s.add_argument("--max-distance", type=int)
    s.set_defaults(func=cmd_cluster)

    s = sub.add_parser("code-clip", help="CLIP zero-shot visual-motif coding (requires [clip] extra)")
    s.add_argument("--model", default="ViT-L-14")
    s.add_argument("--pretrained", default="openai")
    s.add_argument("--thresholds", default="codebook/thresholds.json")
    s.set_defaults(func=cmd_code_clip)

    s = sub.add_parser("sheet", help="export a blank human coding sheet")
    s.add_argument("--out", required=True)
    s.add_argument("--sample", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sheet)

    s = sub.add_parser("import-codes", help="import a completed human coding sheet")
    s.add_argument("sheet")
    s.add_argument("--coder-id", required=True)
    s.set_defaults(func=cmd_import_codes)

    s = sub.add_parser("reliability", help="Krippendorff's alpha across coded sheets")
    s.add_argument("sheets", nargs="+")
    s.set_defaults(func=cmd_reliability)

    s = sub.add_parser("filter", help="apply Filter 1 + Filter 2, write CSV")
    s.add_argument("--out")
    s.add_argument("--only-passing", action="store_true")
    s.set_defaults(func=cmd_filter)

    s = sub.add_parser("schema", help="regenerate JSON Schema")
    s.add_argument("--out", default="schema/image_record.schema.json")
    s.set_defaults(func=cmd_schema)

    s = sub.add_parser("codebook-doc", help="regenerate code tables in docs/CODEBOOK.md")
    s.add_argument("--out", default="docs/CODEBOOK.md")
    s.set_defaults(func=cmd_codebook_doc)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO)
    a.func(a)


if __name__ == "__main__":
    main()
