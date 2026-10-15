# Data Schema (`ImageRecord` v0.1.0)

- **Machine-readable schema:** `schema/image_record.schema.json` (JSON Schema 2020-12), generated from `src/aibi/models.py` with `aibi schema`.
- **Storage:** one JSON object per line in `data/records/<platform>.jsonl`.

## Design rules

1. **One record per image.** `record_id = <platform>:<native_id>:<image_index>`, stable across re-collection.
2. **Append-only evidence.** `ai_evidence`, `engagement`, `motif_codes` and `affect_codes` are lists. Re-collection or re-coding appends entries and never overwrites them, so every value can be traced to a method, coder, version and time.
3. **Raw and standardized side by side.** Engagement snapshots keep the platform's raw fields verbatim (`raw`) next to the common schema (`standard`).
4. **Hashes over bytes.** `sha256`, `phash` and `dhash` identify content without redistributing it. `local_path` exists only where `rights.may_cache_bytes` is true.
5. **No raw personal identifiers.** Authors appear only as `author_pseudonym` (HMAC-SHA256 with a secret project salt).

## Fields

| Path | Type | Notes |
|---|---|---|
| `schema_version` | str | Bumped on breaking change |
| `record_id`, `image_index` | str, int | See rule 1 |
| `source.platform` | str | `reddit`, `bluesky`, `wikimedia_commons`, `diffusiondb` or `factcheck_claimreview` |
| `source.corpus` | `G`/`C`/`V` | Generation, circulation, verified-misuse |
| `source.sampling_frame` | str | `frame_id` from config; links to the run manifest |
| `source.collector`, `collector_version`, `run_id`, `collected_at` | | Run provenance |
| `source.native_id`, `post_url`, `query` | | Where the item came from and which query matched |
| `post.created_at` | datetime (UTC) | Original posting/generation time |
| `post.text`, `alt_text`, `hashtags`, `language`, `community` | | Context; `text` may be omitted where terms require |
| `post.author_pseudonym` | str | `u_` + 24 hex characters |
| `post.platform_labels`, `nsfw` | | Platform-applied labels and flags |
| `image.media_url` | str | Fetch locator. `hf-zip://diffusiondb/part-NNNNNN/<file>` for DiffusionDB |
| `image.sha256`, `phash`, `dhash` | hex | Of the exact bytes fetched |
| `image.mime`, `width`, `height`, `n_bytes`, `fetched_at`, `fetch_error` | | |
| `rights.license`, `terms_basis`, `may_cache_bytes`, `may_redistribute_bytes` | | Legal basis per item |
| `generator`, `prompt` | str | When known (e.g. DiffusionDB) |
| `ai_evidence[]` | `DetectionEvidence` | `criterion` (METHODOLOGY §3), `method`, `value`, `confidence`, `iptc_digital_source_type` |
| `ai_status` | derived | Computed on read, not stored: strongest criterion → `confirmed` / `probable` / `possible` / `unverified` |
| `engagement[]` | `EngagementSnapshot` | `observed_at`, `age_hours`, `raw`, `standard{reactions, amplifications, replies, views, audience}` |
| `motif_codes[]` | `MotifCode` | `code_id`, `present`, `score`, `coder_id`, `coder_type` (`human`/`model`/`consensus`), `codebook_version`, `validated` |
| `affect_codes[]` | `AffectCode` | SAM `valence`/`arousal`/`dominance` (1–9), `primary_emotion`, `secondary_emotion`, `validated` (model coders only) |
| `cluster_id` | str | Near-duplicate cluster (METHODOLOGY §5.4) |
| `exclusion` | str | If set, the record is excluded from analysis, e.g. `lead_unconfirmed`, `csam_suspected_removed`, `user_deleted@<date>` |

## Run manifests

Each manifest is written to `data/runs/<run_id>.json` and contains:

- `run_id`, `started_at` and `finished_at`
- `aibi_version`, `python` and `config_hash`
- `frames[]`: the full frame definition, collector version and counts (`collected`, `fetched`, `fetch_errors`)
- `errors[]`

A manifest plus the codebook version is enough to reproduce or audit any collection.

## Filter output

`aibi filter --out filtered.csv` writes one row per analytic record (`ai_status` ∈ {confirmed, probable}, not excluded). The columns are:

- Identity: `record_id`, `platform`, `corpus`, `sampling_frame`, `ai_status`, `cluster_id`, `post_url`
- Filter 1: `motifs`, `n_motifs`, `n_techniques`, `valence`, `arousal`, `high_affect`, `pai`, `passes_filter1`
- Filter 2: `engagement_total`, `engagement_percentile`, `engagement_rate`, `stratum`, `cluster_n_posts`, `cluster_n_platforms`, `cluster_max_percentile`, `passes_filter2`
