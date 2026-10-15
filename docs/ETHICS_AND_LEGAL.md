# Ethics, Legal and Data-Protection Protocol

This is a working protocol, not legal advice. Before collecting at scale, obtain institutional review (IRB/REC or an exemption determination) and have your institution's counsel or data-protection officer review it.

## 1. Access

- **Official APIs only**, used under their published terms:
  - Reddit Data API (registered app, OAuth)
  - Bluesky / AT Protocol
  - MediaWiki and Wikimedia REST APIs (User-Agent policy, `maxlag`)
  - Google Fact Check Tools API
  - Hugging Face dataset hosting
- **No circumvention:** no login-walled scraping, CAPTCHA solving, rotating proxies or residential IPs, or evasion of rate limits.
- **Non-API pages:** the only non-API fetch is the og:image tag on fact-check pages. These fetches check `robots.txt` (`PoliteSession(respect_robots=True)`).
- **Identification:** every request carries an identifying User-Agent with a contact address. Set it in `config.yaml`.

## 2. Personal data minimization

- Author identifiers are stored only as `HMAC-SHA256(id, AIBI_SALT)`. The salt is kept outside the repository and never published.
- Post text is collected only where it is needed for AI-declaration evidence or context. Public releases drop `post.text` and `alt_text` for platforms whose terms restrict redistribution.
- Coders do not attempt to identify private individuals depicted in images.
- **Deletion compliance.** Before any analysis snapshot and before any release, run `aibi purge`. This follows Reddit's Data API Terms and is good practice everywhere.
  - **What it checks:** Reddit through `/api/info`, Bluesky through `app.bsky.feed.getPosts`, and Wikimedia Commons through `pageids`.
  - **What happens to a deleted or removed item:** its record is turned into a stub (`exclusion: user_deleted@<date>`). Text, image fields, engagement and codes are stripped. Only the record id, source and rights stay, as an audit trail.
  - **Shared files:** cached bytes are deleted unless another live record still uses the same file.
  - **Errors:** an item is stubbed only on positive evidence that it was deleted. Items whose check failed are left unchanged and counted in the report.
  - **Preview:** `--dry-run` reports what would change without changing anything.

## 2a. Sending images to a model-coder API

`aibi code-vlm` sends cached image bytes for coding to the Google Gemini API (default) or the Anthropic API. Before using it:

- **Data-protection basis.** Confirm with your DPO/IRB that third-party processing is covered by your protocol and consent basis. Check your API organization's data-retention configuration.
- **Platform terms.** Confirm that each source platform's terms allow sending its content to a third-party processor for analysis. Reddit's Data API Terms, for example, restrict some uses of content with AI models, so check the current version.
- **When in doubt, don't send.** Exclude a corpus from model coding with `--limit` or a separate data directory, and code it by humans only.
- **API keys.** Keep keys in environment variables (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) or a git-ignored `.env`, never in the repository. Rotate any key that has been pasted into a chat, an issue or a log.
- **Google data terms.** On the Gemini API, the free tier's terms allow Google to use submitted content to improve its products. Use a paid or Vertex AI project for research data.
- **What is sent.** Only image bytes, downscaled to a long edge of at most 1568 px. No captions, author data or URLs are sent.

## 3. Copyright and redistribution

- `rights.may_cache_bytes` and `rights.may_redistribute_bytes` are set per item by the collector.
- **Default public release:** record JSONL with hashes, URLs, codes and engagement. Image bytes are included **only** where `may_redistribute_bytes` is true, for example DiffusionDB (CC0) and freely licensed Commons files.
- Fact-check images are never cached (`may_cache_bytes: false`).

## 4. Harmful content

- **Child sexual abuse material (CSAM).** If an image appears to depict a minor sexually, stop immediately. Do not view further, save, screenshot or share it.
  - Record only the URL, then delete any cached bytes and set `exclusion: csam_suspected_removed`.
  - Report it to NCMEC (CyberTipline) in the US, the IWF in the UK, or the equivalent national hotline (INHOPE member).
  - Notify the PI.
- **Sexualized real people (VM19).** Store hashes and URLs only, never bytes.
- **Automated flags.** The model coder returns a `safety_flag` for every image (`aibi/safety.py`).
  - `possible_minor_sexualization`: cached bytes are deleted immediately, the record is excluded (`csam_suspected_removed`), and a critical log line gives the URL so a person can follow the reporting steps above.
  - `sexualized_real_person`: bytes are deleted, following the VM19 handling rule.
  - Automated flags can be false positives. A deleted CC0 image can be fetched again after human review.
- **Swipe app protections.** Images are blurred until tapped by default. Every card has a ⚠ report button, which hides the image for all coders. "May sexualize a minor" also deletes the stored file and logs the URL for reporting. The app is served with an access token and is meant for a trusted local network.
- **Coder wellbeing.** Propaganda corpora include gore, dehumanization and hate imagery. Rules for coders:
  - Sessions are capped at 90 minutes.
  - Coders may skip an item without penalty.
  - Gore and hate content gets a content warning in coding sheets.
  - Debriefing and support resources are available.
- **Hate content.** Hate imagery is studied, not amplified. Public outputs show such images only where scholarly necessity justifies it, and with context.

## 5. Dual-use

A ranked list of high-engagement, high-affect AI propaganda images could be misused as a playbook. Two safeguards apply:

- Public releases report aggregate findings and codebook-level patterns.
- Item-level data on the top engagement tier goes to vetted researchers under a data-use agreement.
