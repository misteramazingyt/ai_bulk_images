"""Embedded-provenance inspection of image bytes (criteria C2PA / META).

Most platforms strip metadata, so absence of evidence here is never treated
as evidence of non-AI origin (METHODOLOGY §3).
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile

from PIL import Image

from .models import Criterion, DetectionEvidence

IPTC_DST = "http://cv.iptc.org/newscodes/digitalsourcetype/"
AI_DST_TERMS = ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")

# PNG tEXt/iTXt keys written by common open generators.
GENERATOR_PNG_KEYS = {
    "parameters": "AUTOMATIC1111/Forge-style parameters",
    "prompt": "generator prompt chunk (ComfyUI graph or plain prompt)",
    "workflow": "ComfyUI workflow",
    "invokeai_metadata": "InvokeAI",
    "sd-metadata": "InvokeAI (legacy)",
    "Dream": "lstein/dream",
}
GENERATOR_SOFTWARE = re.compile(
    r"(stable diffusion|midjourney|dall[-·\s]?e|firefly|imagen|ideogram|leonardo|"
    r"novelai|comfyui|automatic1111|invokeai|flux|gpt-?4o|sora|bing image creator)",
    re.I,
)
_XMP_RE = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)


def _xmp_bytes(raw: bytes, img: Image.Image) -> bytes:
    xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
    if isinstance(xmp, str):
        return xmp.encode("utf-8", "ignore")
    if isinstance(xmp, bytes):
        return xmp
    m = _XMP_RE.search(raw)
    return m.group(0) if m else b""


def has_c2pa_container(raw: bytes) -> bool:
    """Heuristic presence check for a C2PA JUMBF store (JPEG APP11 / PNG caBX)."""
    return b"c2pa" in raw[:4_000_000] and (b"jumb" in raw or b"caBX" in raw)


def validate_c2pa(raw: bytes) -> dict | None:
    """Validate with the official `c2patool` CLI if installed; else None."""
    tool = shutil.which("c2patool")
    if not tool:
        return None
    with tempfile.NamedTemporaryFile(suffix=".img") as f:
        f.write(raw)
        f.flush()
        proc = subprocess.run([tool, f.name], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {"valid": False, "error": proc.stderr.strip()[:500]}
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"valid": False, "error": "unparseable c2patool output"}
    has_errors = bool(report.get("validation_status"))
    return {"valid": not has_errors, "report": report}


def inspect_bytes(raw: bytes) -> list[DetectionEvidence]:
    evidence: list[DetectionEvidence] = []
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return evidence

    for key, label in GENERATOR_PNG_KEYS.items():
        if key in img.info and isinstance(img.info[key], str) and img.info[key].strip():
            evidence.append(
                DetectionEvidence(
                    criterion=Criterion.META,
                    method=f"png_text_chunk:{key}",
                    value=label,
                    iptc_digital_source_type=IPTC_DST + "trainedAlgorithmicMedia",
                )
            )

    xmp = _xmp_bytes(raw, img)
    for term in AI_DST_TERMS:
        if term.encode() in xmp:
            evidence.append(
                DetectionEvidence(
                    criterion=Criterion.META,
                    method="xmp:Iptc4xmpExt:DigitalSourceType",
                    value=term,
                    iptc_digital_source_type=IPTC_DST + term,
                )
            )
            break

    try:
        software = img.getexif().get(0x0131)
    except Exception:
        software = None
    if isinstance(software, str) and GENERATOR_SOFTWARE.search(software):
        evidence.append(DetectionEvidence(criterion=Criterion.META, method="exif:Software", value=software[:200]))

    if has_c2pa_container(raw):
        result = validate_c2pa(raw)
        manifest = json.dumps(result.get("report", {})) if result else ""
        ai_term = next((t for t in AI_DST_TERMS if t in manifest), None)
        if result and result.get("valid") and ai_term:
            evidence.append(
                DetectionEvidence(
                    criterion=Criterion.C2PA,
                    method="c2patool:validated",
                    value=ai_term,
                    iptc_digital_source_type=IPTC_DST + ai_term,
                )
            )
        else:
            # Present but not validated (or validation unavailable): weaker META.
            evidence.append(
                DetectionEvidence(
                    criterion=Criterion.META,
                    method="c2pa:present_unvalidated",
                    value=ai_term or ("validation_failed" if result else "c2patool_unavailable"),
                )
            )
    return evidence


def text_declares_ai(text: str | None) -> str | None:
    """Return the matched phrase if free text (caption/alt/hashtags) declares AI origin."""
    if not text:
        return None
    m = DECLARATION_RE.search(text)
    return m.group(0) if m else None


DECLARATION_RE = re.compile(
    r"(#?ai[\s_-]?(generated|art|artwork|image|imagery)\b|#?midjourney|#?stablediffusion|stable diffusion|"
    r"#?dall[-\s]?e\s?\d?|#?aiart\w*|made (with|using|by) (ai|midjourney|dall-?e|firefly|flux)|"
    r"#?generativeai|#flux\w*|flux\.1|#?sdxl|#?comfyui|#ideogram\w*|#adobefirefly|#firefly(ai)?\b)",
    re.I,
)
