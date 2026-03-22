from __future__ import annotations

import json
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent / "datasets"
LANGUAGE_CORPORA = {
    "korean": DATASET_ROOT / "korean_sentences.json",
}


def _load_sentences(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sentences = payload.get("sentences", [])
    if not isinstance(sentences, list) or not all(isinstance(item, str) for item in sentences):
        raise ValueError(f"Invalid text corpus format: {path}")
    return sentences


def generate_korean_sentences(target_count: int) -> list[str]:
    sentences = _load_sentences(LANGUAGE_CORPORA["korean"])
    if target_count > len(sentences):
        raise ValueError(
            f"Requested {target_count} Korean prompts but only {len(sentences)} are available in the corpus."
        )
    return sentences[:target_count]


def generate_sentences(language: str, target_count: int) -> list[str]:
    normalized = language.strip().lower()
    if normalized == "korean":
        return generate_korean_sentences(target_count)
    raise ValueError(f"Unsupported generation language: {language}")
