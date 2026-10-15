# ai_bulk_images

This project collects AI-generated images from relatively open sources and documents every step. It then filters them on two axes:

1. **Propagandistic and high-affect visual motifs**, coded with a standardized four-layer codebook:
   - 20 visual motifs
   - the 22 persuasion techniques of SemEval-2021 Task 6
   - SAM valence, arousal and dominance ratings
   - a discrete target emotion
2. **Presence and engagement across social media**, measured with:
   - a common engagement schema
   - T+7d snapshots
   - within-stratum percentiles
   - cross-platform near-duplicate clusters (pHash)

## Documentation

Read these before collecting anything.

| Doc | Contents |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Research design, corpora and sampling frames, AI-detection criteria, filters, normalization, reliability, limitations, references |
| [`docs/CODEBOOK.md`](docs/CODEBOOK.md) | Coding manual and code tables. The tables are generated from [`codebook/motifs.yaml`](codebook/motifs.yaml) |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | The `ImageRecord` schema, with the JSON Schema at [`schema/image_record.schema.json`](schema/image_record.schema.json) |
| [`docs/ETHICS_AND_LEGAL.md`](docs/ETHICS_AND_LEGAL.md) | Access rules, pseudonymization, copyright, harmful-content protocol |
| [`docs/AMENDMENTS.md`](docs/AMENDMENTS.md) | Versioned log of every change to rules or thresholds |

## Sources (v0.1)

| Corpus | Source | Credentials |
|---|---|---|
| G (generation) | DiffusionDB 2M (CC0), read from remote zips with HTTP Range requests | none |
| C (circulation) | Reddit Data API | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` |
| C | Bluesky (AT Protocol) | `BSKY_HANDLE`, `BSKY_APP_PASSWORD` (an app password) |
| C | Wikimedia Commons, *AI-generated images* category tree, plus pageviews | none (set a User-Agent with contact info) |
| V (verified misuse) | Google Fact Check Tools (ClaimReview) | `GOOGLE_FACTCHECK_API_KEY` |

All sources also need `AIBI_SALT`, a long random secret used for author pseudonymization.

## Quick start

```bash
pip install -e ".[dev]"          # add ".[clip]" for the CLIP zero-shot coder (installs torch)
cp config.example.yaml config.yaml   # edit the contact User-Agent and the frames
export AIBI_SALT="$(python -c 'import secrets; print(secrets.token_hex(32))')"   # store it safely and reuse it

aibi collect --frame G-diffusiondb-pilot   # writes data/records/*.jsonl and data/runs/<run_id>.json
aibi cluster                               # near-duplicate clusters across platforms
aibi sheet --out coderA.csv --sample 200   # blank human coding sheet
aibi import-codes coderA.csv --coder-id A
aibi reliability coderA.csv coderB.csv     # Krippendorff's alpha per variable
aibi code-ui --sample 200                  # swipe-to-code web app for human coders' phones (see below)
aibi ui-export                             # coder logs -> data/coding/sheets/<coder>.csv
aibi code-vlm --limit 20                   # model coder; Gemini by default (pip install -e ".[gemini]"; GEMINI_API_KEY)
aibi code-vlm --provider anthropic --batch # Claude through the Message Batches API at 50% cost (".[vlm]"); prints a batch_id
aibi vlm-collect <batch_id> --wait --retry-sync
aibi calibrate --coder-id <id> --write --apply   # validate against human gold; writes codebook/thresholds.json
aibi purge                                 # remove items deleted at source; run before analysis or release
aibi filter --out filtered.csv             # Filter 1 + Filter 2 metrics per image
```

Model-coder codes count only after `aibi calibrate` validates them (METHODOLOGY §4.3). Read `docs/ETHICS_AND_LEGAL.md` §2a before sending images to an API.

## Hand coding on a phone

`aibi code-ui` prints a link containing an access token, e.g. `http://192.168.1.20:8765/?t=…`. Open it on any phone on the same network.

- **Answering.** Each card shows one image and one code. Swipe right for present, left for absent, up for unsure. Buttons and the arrow keys do the same on a laptop.
  - **Order:** the default "one code at a time" order runs a single code across every image.
  - **Affect pass:** a separate pass for the 1–9 valence and arousal ratings and the target emotion, entered by tapping.
- **What coders see.** Images are blurred until tapped. Coders never see the caption, platform or engagement numbers.
- **Undo and resuming.** Undo always works, and coders can resume where they left off. Every decision is appended to `data/coding/<coder>.jsonl`.
- **Report button (⚠).** Hides an image for everyone. Choosing "may sexualize a minor" also deletes the file immediately.
- **Same sample for everyone.** Every coder gets the same seeded sample, so the results can be used for double-coding. `aibi ui-export` writes standard coding sheets for `aibi import-codes` and `aibi reliability`.
- **Remote coders.** The server is plain HTTP and is meant for a trusted local network. For coders elsewhere, put it behind a VPN or an HTTPS tunnel instead of exposing the port.

## Development

```bash
python -m pytest            # unit tests, offline
aibi schema                 # regenerate the JSON Schema after editing models.py
aibi codebook-doc           # regenerate the CODEBOOK.md tables after editing motifs.yaml
```

## Layout

```
codebook/motifs.yaml      standardized codebook (versioned)
schema/                   generated JSON Schema
src/aibi/
  models.py               ImageRecord and evidence/code/engagement models
  provenance.py           C2PA / XMP IPTC DigitalSourceType / PNG generator-chunk inspection
  hashing.py              SHA-256, pHash/dHash, exact banded near-duplicate clustering
  engagement.py           platform -> common schema, T+7d selection, stratified percentiles
  reliability.py          Krippendorff's alpha, threshold calibration, AUROC
  filters.py              Filter 1 (motif + affect) and Filter 2 (spread + engagement)
  collectors/             diffusiondb, reddit, bluesky, wikimedia, factcheck
  coders/                 human sheet round-trip, CLIP zero-shot, vision-language model (sync + batch)
  calibration.py          model-coder validation -> thresholds.json
  safety.py, purge.py     harmful-content flags, deletion compliance
  pipeline.py, cli.py
```
