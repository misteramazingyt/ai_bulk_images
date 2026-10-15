"""Load, validate and render the motif codebook (codebook/motifs.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "codebook" / "motifs.yaml"


@dataclass(frozen=True)
class Code:
    id: str
    name: str
    definition: str
    layer: str  # "visual_motif" | "persuasion_technique"
    modality: str = "image"
    include: str | None = None
    exclude: str | None = None
    lexicon: tuple[str, ...] = ()
    clip_prompts: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Codebook:
    version: str
    instructions: str
    codes: dict[str, Code]
    emotions: list[str]
    affect: dict[str, Any]
    filters: dict[str, Any]

    @property
    def visual_motifs(self) -> list[Code]:
        return [c for c in self.codes.values() if c.layer == "visual_motif"]

    @property
    def techniques(self) -> list[Code]:
        return [c for c in self.codes.values() if c.layer == "persuasion_technique"]

    def lexicon_matches(self, text: str | None) -> list[str]:
        """Code ids whose lexicon terms occur in text (pre-screening only, §2.2)."""
        if not text:
            return []
        t = text.lower()
        hits = []
        for c in self.visual_motifs:
            if any(re.search(r"\b" + re.escape(term.lower()) + r"\b", t) for term in c.lexicon):
                hits.append(c.id)
        return hits


_KNOWN = {"id", "name", "definition", "modality", "include", "exclude", "lexicon", "clip_prompts"}


def _code(d: dict, layer: str) -> Code:
    for k in ("id", "name", "definition"):
        if not d.get(k):
            raise ValueError(f"codebook entry missing '{k}': {d}")
    return Code(
        id=d["id"],
        name=d["name"],
        definition=d["definition"],
        layer=layer,
        modality=d.get("modality", "image"),
        include=d.get("include"),
        exclude=d.get("exclude"),
        lexicon=tuple(str(t) for t in d.get("lexicon") or ()),
        clip_prompts=tuple(d.get("clip_prompts") or ()),
        extra={k: v for k, v in d.items() if k not in _KNOWN},
    )


@lru_cache(maxsize=4)
def load(path: str | Path = DEFAULT_PATH) -> Codebook:
    data = yaml.safe_load(Path(path).read_text())
    codes: dict[str, Code] = {}
    for layer_key, layer in (("visual_motifs", "visual_motif"), ("persuasion_techniques", "persuasion_technique")):
        for entry in data.get(layer_key, []):
            c = _code(entry, layer)
            if c.id in codes:
                raise ValueError(f"duplicate code id {c.id}")
            if c.modality not in ("image", "text", "both"):
                raise ValueError(f"{c.id}: bad modality {c.modality}")
            codes[c.id] = c
    return Codebook(
        version=str(data["codebook_version"]),
        instructions=data.get("coding_instructions", "").strip(),
        codes=codes,
        emotions=list(data.get("discrete_emotions", [])),
        affect=data.get("affect", {}),
        filters=data.get("filters", {}),
    )


def render_markdown(cb: Codebook) -> str:
    """Render the code tables for docs/CODEBOOK.md (keeps doc and YAML in sync)."""
    esc = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = [f"## Code tables (generated from `codebook/motifs.yaml`, v{cb.version})", "",
             "### Layer 1 - Visual motifs", "",
             "| ID | Name | Definition | Include | Exclude |", "|---|---|---|---|---|"]
    for c in cb.visual_motifs:
        lines.append(f"| {c.id} | `{c.name}` | {esc(c.definition)} | {esc(c.include)} | {esc(c.exclude)} |")
    lines += ["", "### Layer 2 - Persuasion techniques (SemEval-2021 Task 6)", "",
              "| ID | Name | Modality | Definition |", "|---|---|---|---|"]
    for c in cb.techniques:
        lines.append(f"| {c.id} | `{c.name}` | {c.modality} | {esc(c.definition)} |")
    lines += ["", "### Layer 3 - Affect (SAM, 1-9)", ""]
    for dim, spec in cb.affect.get("dimensions", {}).items():
        anchors = "; ".join(f"{k} = {v}" for k, v in spec.get("anchors", {}).items())
        lines.append(f"- **{dim}**{' (optional)' if spec.get('optional') else ''}: {anchors}")
    lines += ["", "### Layer 4 - Discrete target emotion", "", ", ".join(f"`{e}`" for e in cb.emotions), ""]
    return "\n".join(lines)
