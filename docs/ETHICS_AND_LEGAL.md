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
- **Deletion compliance.** Before any analysis snapshot and before any release, items deleted at the source are removed. This follows Reddit's Data API Terms and is good practice everywhere. The procedure: re-check each `post_url` and set `exclusion: user_deleted`.

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
