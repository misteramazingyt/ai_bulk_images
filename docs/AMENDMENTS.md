# Amendments Log

Record every change to an inclusion rule, filter definition, default threshold or codebook definition.

| Date | Version | Change | Rationale | Affects existing data? |
|---|---|---|---|---|
| 2026-10-15 | 0.1.0 | Initial research design, codebook and schema | — | — |
| 2026-10-15 | 0.1.0 | Lexicon pre-screen: dropped "crying", "begging" (VM03) and bare "poster", "meme", "caption" (VM15) | Pilot run on DiffusionDB showed generic style tokens over-matching | No (pre-screen only affects sampling; no data collected yet) |
| 2026-10-15 | 0.1.0 | Gemini added as the default model-coder provider; swipe app for human coding (unsure = blank in sheets) | Provider choice; faster human coding | No |
| 2026-10-15 | 0.1.0 | Model-coder validation also requires ≥ 10 gold positives in the test half; VLM coder and `AffectCode.validated` added | F1 on very few positives is unstable | No |
