"""CLIP zero-shot visual-motif coder (optional extra: `pip install -e .[clip]`).

Score for motif m = max over m's positive prompts of the softmax probability
against [positives + shared neutral prompts]. Scores are thresholded using
codebook/thresholds.json; a code is marked `validated` only if the threshold
file says the motif passed METHODOLOGY §4.3 for this coder_id.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from ..codebook import Codebook
from ..models import MotifCode

NEUTRAL_PROMPTS = [
    "an ordinary everyday photo", "a landscape photograph", "a portrait of a person",
    "a colorful abstract digital artwork", "a fantasy illustration", "a product photo", "a cute animal",
]


class ClipZeroShotCoder:
    def __init__(self, codebook: Codebook, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 thresholds_path: str | Path = "codebook/thresholds.json", device: str | None = None):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.cb = codebook
        self.coder_id = f"clip-zeroshot:{model_name}:{pretrained}:cb{codebook.version}"
        self.motifs = [m for m in codebook.visual_motifs if m.clip_prompts]
        tp = Path(thresholds_path)
        self.thresholds = json.loads(tp.read_text()).get(self.coder_id, {}) if tp.exists() else {}
        with torch.no_grad():
            self._text = {}
            for m in self.motifs:
                prompts = list(m.clip_prompts) + NEUTRAL_PROMPTS
                emb = self.model.encode_text(self.tokenizer(prompts).to(self.device))
                self._text[m.id] = (emb / emb.norm(dim=-1, keepdim=True), len(m.clip_prompts))

    def score(self, raw: bytes) -> dict[str, float]:
        from PIL import Image

        img = self.preprocess(Image.open(io.BytesIO(raw)).convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            v = self.model.encode_image(img)
            v = v / v.norm(dim=-1, keepdim=True)
            out = {}
            for mid, (t, n_pos) in self._text.items():
                probs = (100.0 * v @ t.T).softmax(dim=-1)[0]
                out[mid] = float(probs[:n_pos].max())
        return out

    def code(self, raw: bytes) -> list[MotifCode]:
        codes = []
        for mid, s in self.score(raw).items():
            th = self.thresholds.get(mid, {})
            codes.append(MotifCode(
                code_id=mid, present=s >= th.get("threshold", 0.5), score=s, coder_id=self.coder_id,
                coder_type="model", codebook_version=self.cb.version, validated=bool(th.get("validated")),
                note=None if th else "uncalibrated default threshold 0.5",
            ))
        return codes
