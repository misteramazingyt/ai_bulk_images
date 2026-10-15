# Methodology

**Project:** documented collection and filtration of AI-generated images from open sources, filtered for (1) propagandistic / high-affect visual motifs and (2) presence and engagement across social media.

**Status:** v0.1 research design (first pass). Every threshold marked *default* is a pre-registered starting value. Change it only through a versioned amendment (§9), and report sensitivity analyses for it.

---

## 1. Research questions

- **RQ1. Prevalence.** Among AI-generated images in open corpora, what share shows codebook-defined propagandistic or high-affect motifs?
- **RQ2. Circulation.** Do images with those motifs reach systematically higher normalized engagement, or wider cross-platform presence, than AI images without them?
- **RQ3. Composition.** Which motifs (e.g. vulnerable victims, outgroup dehumanization, leader glorification or degradation) and which affect profiles (valence × arousal) are over-represented among high-engagement images?

## 2. Corpora and sampling frames

There are three corpora. Every record carries a `sampling_frame` id, so the corpora are never silently pooled.

| Corpus | Purpose | Sources (v0.1) | AI-status basis |
|---|---|---|---|
| **G — Generation-side** | What people *make* with generators (base rates) | DiffusionDB 2M (CC0) | `DATASET` |
| **C — Circulation-side** | What *circulates* and how much engagement it gets | Reddit Data API, Bluesky (AT Protocol), Wikimedia Commons | `DECLARED`, `PLATFORM`, `META` |
| **V — Verified-misuse** | AI images that fact-checkers have documented as deceptive | Google Fact Check Tools API (ClaimReview) | `FACTCHECK` |

Restricted platforms (X, Meta, TikTok, Telegram) are **out of scope for v0.1**. If access is later obtained through sanctioned research programmes (e.g. Meta Content Library, the TikTok Research API), they get their own adapters and sampling frames.

### 2.1 Sampling-frame definition (required for every collection run)

Each frame is declared in `config.yaml` and frozen in the run manifest (`data/runs/<run_id>.json`). A frame consists of:

- `frame_id`, `source`, `corpus` (G/C/V)
- **Query set:** exact search terms, hashtags, subreddits, categories, or dataset filter expressions
- **Time window:** `since` / `until` (UTC, ISO-8601)
- **Ordering:** e.g. `new`, `top`, or `latest`. API "top/hot" orderings are *engagement-selected* and therefore biased for RQ2. Use them only for descriptive work, never for inferential comparisons.
- **Sample size and seed:** random sub-sampling uses a fixed `seed`
- **Software version and config hash:** written automatically

### 2.2 Pre-screening vs. coding

For large text-indexed corpora (DiffusionDB prompts, post captions), the pipeline may *pre-screen* with the codebook lexicons (`codebook/motifs.yaml`, `lexicon:` fields) to oversample likely-positive images.

- Pre-screening is a **sampling device only, never a coding decision.**
- Each pre-screened stratum must be paired with a random comparison sample from the same frame, and analyses must use the stratum weights stored in the manifest.

## 3. AI-generation criteria (standardized)

An image enters the corpus only with recorded evidence that it is AI-generated. Evidence is coded under a fixed set of **criterion codes**. Each one aligns with the IPTC *Digital Source Type* controlled vocabulary (`http://cv.iptc.org/newscodes/digitalsourcetype/`) where applicable.

| Code | Evidence | Example | Strength |
|---|---|---|---|
| `C2PA` | Validated C2PA manifest asserting `digitalSourceType` = `trainedAlgorithmicMedia` or `compositeWithTrainedAlgorithmicMedia` | Images exported from generators that sign Content Credentials | confirmed |
| `META` | Unsigned embedded generator metadata: IPTC DST in XMP, generator parameters in PNG text chunks (`parameters`, `prompt`, `workflow`), a known generator in EXIF `Software`, or a C2PA manifest that is present but unvalidated | A1111/ComfyUI PNGs | confirmed |
| `DATASET` | The image comes from a dataset built by logging generations at the point of creation | DiffusionDB | confirmed |
| `FACTCHECK` | A recognized fact-checker (an IFCN signatory publishing ClaimReview) explicitly found the image to be AI-generated | "Pentagon explosion" image, May 2023 | confirmed |
| `PLATFORM` | Platform- or curator-applied AI label | Wikimedia Commons *AI-generated images* category tree | probable |
| `DECLARED` | The creator or venue declared it AI-generated: hashtag (`#aiart`, `#midjourney`), caption or alt-text, or an AI-only community whose rules require AI content (r/midjourney) | Reddit, Bluesky | probable |
| `DETECTOR` | A statistical AI-image detector score at or above its calibrated threshold | — | possible |

**Derived `ai_status`**, which is the strongest status any single piece of evidence supports:

- `confirmed`: any of `C2PA`, `META`, `DATASET`, `FACTCHECK`
- `probable`: any of `PLATFORM`, `DECLARED`
- `possible`: `DETECTOR` only
- `unverified`: no evidence

**Inclusion rule (default):** the analytic corpus contains only `confirmed` and `probable` images. `possible` images are retained but kept out of the analytic corpus. Detector scores are never the sole basis for inclusion, because detector false-positive rates on real photographs are poorly characterized and degrade under recompression.

**Absence of metadata is uninformative.** Most platforms strip EXIF, XMP and C2PA metadata on upload. Missing metadata is never treated as evidence that an image is *not* AI-generated.

## 4. Motif coding (standardized codebook)

The full codebook is in `docs/CODEBOOK.md`; the machine-readable version is `codebook/motifs.yaml`, with a `codebook_version` stamped on every code. It has four layers, each grounded in an existing published instrument:

1. **Visual motifs (VM01–VM20).** These are denotative and observable: *what is depicted*. This is the primary unit for filtering because concrete content codes reliably. The motifs are drawn from the visual-propaganda literature and framed with Rodriguez & Dimitrova's (2011) four-tier visual framing model, coding at the denotative and stylistic tiers.
2. **Persuasion techniques (PT01–PT22).** These are connotative: *what the image does rhetorically*. They are adopted from the 22-technique taxonomy of SemEval-2021 Task 6, *Detection of Persuasion Techniques in Texts and Images* (Dimitrov et al., 2021), which builds on Da San Martino et al. (2019). Each technique is flagged `image`, `text` or `both`, and text-only techniques are coded only when overlay text is present.
3. **Affect.** Ratings follow the Self-Assessment Manikin (SAM; Bradley & Lang, 1994): valence, arousal and dominance, each on a 9-point scale. Coders rate the **response the image is designed to elicit in a general adult viewer**, not their own response. This is the convention for intended-affect coding in political communication.
4. **Discrete target emotion.** Coders choose one primary emotion and optionally one secondary emotion from {anger, fear, disgust, contempt, sadness, pride, joy, awe, none}. The set covers Ekman's basic emotions plus the moral and political emotions (pride, contempt, awe) prominent in the propaganda literature.

### 4.1 Coders

- **Human coders.** At least two trained coders, each with a documented codebook-training session that ends when they pass a 50-image training set at α ≥ 0.80.
- **Model coders.** Automated coders are registered with a stable `coder_id`: CLIP zero-shot (`coders/clip_zeroshot.py`), or a vision-language model prompted with the codebook definitions verbatim. A model coder's output counts as a code **only after validation against human gold labels** (§4.3).

### 4.2 Reliability

- **Double-coding sample.** At least 10% of the analytic corpus and no fewer than 200 images. It is stratified by corpus and drawn with the manifest seed.
- **Statistic.** Krippendorff's α (`aibi reliability`): nominal α for motif presence and discrete emotion, interval α for SAM ratings.
- **Thresholds** (Krippendorff, 2004): α ≥ 0.800 → reliable, and the variable is used in analysis. 0.667 ≤ α < 0.800 → tentative conclusions only, flagged in tables. α < 0.667 → the variable is dropped or redefined through a codebook amendment.

### 4.3 Model-coder validation

For each motif, compute precision, recall, F1 and AUROC against a held-out human gold set of at least 50 positives where they are obtainable. Report the confusion matrix.

The threshold per motif is chosen on a *calibration* split (default rule: maximize F1, or the lowest threshold giving precision ≥ 0.80) and then evaluated on a separate *test* split. The thresholds are stored in `codebook/thresholds.json`, keyed by `coder_id` and `codebook_version`.

A motif whose model coder has test-set F1 < 0.70 is coded **by humans only**.

### 4.4 Filter 1: propagandistic / high-affect (default definition)

An image **passes Filter 1** if the consensus codes (human adjudication, or a validated model coder) contain:

- ≥ 1 **visual motif** from VM01–VM20, **or** ≥ 1 **persuasion technique of modality `image` or `both`**,

**and** the image is **high-affect**:

- SAM arousal ≥ 7, **or**
- arousal ≥ 6 **and** |valence − 5| ≥ 3.

The pipeline also records a continuous **Propaganda-Affect Index**:

`PAI = n_motifs_present + n_techniques_present + (arousal − 1)/8 + |valence − 5|/4`

Report PAI alongside the binary filter. Sensitivity analyses vary the arousal cut-off between 6 and 8.

## 5. Engagement measurement (standardized)

### 5.1 Common engagement schema

Raw platform metrics are always kept verbatim in `engagement.raw`. They are also mapped to four standardized components:

| Component | Meaning | Reddit | Bluesky | Wikimedia Commons |
|---|---|---|---|---|
| `reactions` | Low-effort endorsement | `score` (net votes, see note) | `likeCount` | — |
| `amplifications` | Redistribution to new audiences | `num_crossposts` | `repostCount + quoteCount` | — |
| `replies` | Conversational response | `num_comments` | `replyCount` | — |
| `views` | Exposure | — (not exposed by the API) | — | pageviews (REST API) |
| `audience` | Potential reach at observation time | `subreddit_subscribers` | author `followersCount` | — |

**Reddit note:** `score` is upvotes minus downvotes and is fuzzed by Reddit. `upvote_ratio` is kept in `raw`. Reddit `reactions` are therefore not strictly comparable to likes, which is one reason the primary cross-platform metric is a within-platform percentile.

### 5.2 Observation timing

Engagement accrues over time, so every snapshot stores `observed_at` and `age_hours` (time since the post was created).

- **Primary analysis** uses snapshots with `age_hours` ≥ 168 (**T+7d**), the age by which engagement on these platforms has largely saturated.
- Snapshots taken earlier are retained, but only for growth curves.
- Re-collection schedule (default): T+24h, T+7d and T+30d where feasible.

### 5.3 Normalization

Implemented in `aibi/engagement.py`:

1. **Total engagement:** `E = reactions + amplifications + replies` (each component also analysed separately).
2. **Engagement rate:** `ER = E / audience` when audience is known.
3. **Within-stratum percentile (primary cross-platform metric):** the percentile rank of `E` within its **stratum**, which is `platform × community (subreddit/feed or none) × calendar month`, using the T+7d snapshot. A stratum needs n ≥ 30 posts; smaller strata fall back to `platform × month`.
4. **Within-stratum z-score** of `log1p(E)`, as a secondary metric.
5. **High engagement (default):** within-stratum percentile ≥ 90.

### 5.4 Filter 2: cross-platform presence and engagement

The unit of analysis for presence is the **image cluster**:

- Exact duplicates are grouped by SHA-256.
- Near-duplicates (resized, recompressed, lightly cropped) are grouped by a 64-bit perceptual hash (pHash), linking images at Hamming distance ≤ 8 (*default*) using single-linkage union-find.
- Thresholds above 8 inflate false merges. Cluster purity is validated on a manual sample of 100 clusters, reporting the false-merge rate.

Per cluster, the pipeline computes `n_posts`, `n_platforms`, `first_seen`, `max_percentile` and `sum_E`.

An image **passes Filter 2** if its cluster has ≥ 1 social post with a T+7d snapshot, **and** either `max_percentile` ≥ 90 **or** `n_platforms` ≥ 2.

## 6. Data schema and provenance

Every item is one `ImageRecord` (`docs/DATA_SCHEMA.md`, `schema/image_record.schema.json`, generated from `aibi/models.py`). Provenance is captured at four levels:

1. **Item:** source URL, native ID, `collected_at`, collector name and version, sampling frame, query that matched.
2. **Content:** SHA-256 and pHash/dHash of the exact bytes retrieved, plus MIME type and dimensions. Hashes let other researchers verify and re-identify items without redistributing images.
3. **Evidence:** every AI-detection and motif code is a separate entry with its method, coder, version and timestamp. Codes are appended, never overwritten.
4. **Run:** a manifest with config hash, package version, frame definitions, counts and errors.

## 7. Ethics, legal and data-protection

See `docs/ETHICS_AND_LEGAL.md`. In brief:

- Use official APIs only and follow their terms.
- No scraping that circumvents access controls.
- Author identifiers are stored only as keyed hashes (HMAC-SHA256).
- Images are cached locally only where the license or terms allow, and the default releasable dataset is hashes plus URLs plus codes.
- Platform deletion-compliance requests are honoured.
- Any suspected CSAM is excluded immediately, never stored, and reported.

## 8. Known limitations

- **Selection bias in circulation-side corpora.** Self-declared AI images over-represent hobbyist art and under-represent covert propaganda, which by design is *not* labelled. Corpus V partially compensates.
- **Engagement metrics are not exposure.** No v0.1 platform exposes impressions for posts.
- **Detection asymmetry.** We can confirm AI origin but rarely refute it.
- **Temporal drift.** Generator styles change quickly. Codebook motifs are content-based and generator-agnostic, but model-coder validation must be repeated for each corpus period.

## 9. Versioning and amendments

- Codebook, thresholds and this document are versioned in git.
- Any change to an inclusion rule, filter definition or default threshold gets an entry in `docs/AMENDMENTS.md` with the date, the rationale, and whether data collected before the change is affected.

## References

- Bradley, M. M., & Lang, P. J. (1994). Measuring emotion: The Self-Assessment Manikin and the semantic differential. *Journal of Behavior Therapy and Experimental Psychiatry, 25*(1), 49–59.
- C2PA. *Content Credentials: C2PA Technical Specification* (v2.x). https://c2pa.org/specifications/
- Da San Martino, G., Yu, S., Barrón-Cedeño, A., Petrov, R., & Nakov, P. (2019). Fine-grained analysis of propaganda in news articles. *EMNLP-IJCNLP*.
- Dimitrov, D., Bin Ali, B., Shaar, S., Alam, F., Silvestri, F., Firooz, H., Nakov, P., & Da San Martino, G. (2021). SemEval-2021 Task 6: Detection of persuasion techniques in texts and images. *SemEval-2021*.
- IPTC. *Digital Source Type NewsCodes vocabulary.* http://cv.iptc.org/newscodes/digitalsourcetype/
- Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2nd ed.). Sage.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML*.
- Rodriguez, L., & Dimitrova, D. V. (2011). The levels of visual framing. *Journal of Visual Literacy, 30*(1), 48–65.
- Wang, Z. J., Montoya, E., Munechika, D., Yang, H., Hoover, B., & Chau, D. H. (2023). DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models. *ACL*.
- Zauner, C. (2010). *Implementation and benchmarking of perceptual image hash functions* (pHash). Master's thesis, Upper Austria University of Applied Sciences.
