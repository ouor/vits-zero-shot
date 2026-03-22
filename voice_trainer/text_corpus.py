from __future__ import annotations

import json
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent / "datasets"
CORPUS_PATH = DATASET_ROOT / "multilingual_sentences.json"
LANGUAGE_ALIASES = {
    "ko": "korean",
    "korean": "korean",
    "kr": "korean",
    "en": "english",
    "english": "english",
    "jp": "japanese",
    "ja": "japanese",
    "japanese": "japanese",
    "cn": "chinese",
    "zh": "chinese",
    "chinese": "chinese",
}


def _load_corpora(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    corpora = payload.get("languages", {})
    if not isinstance(corpora, dict):
        raise ValueError(f"Invalid text corpus format: {path}")
    for language, sentences in corpora.items():
        if not isinstance(language, str) or not isinstance(sentences, list):
            raise ValueError(f"Invalid text corpus format: {path}")
        if not all(isinstance(item, str) for item in sentences):
            raise ValueError(f"Invalid text corpus format: {path}")
    return corpora


def _normalize_language(language: str) -> str:
    normalized = language.strip().lower()
    canonical = LANGUAGE_ALIASES.get(normalized, normalized)
    if canonical not in _load_corpora(CORPUS_PATH):
        raise ValueError(f"Unsupported generation language: {language}")
    return canonical


def _generate_sentences_for_language(language: str, target_count: int) -> list[str]:
    corpora = _load_corpora(CORPUS_PATH)
    sentences = corpora[language]
    if target_count > len(sentences):
        raise ValueError(
            f"Requested {target_count} {language} prompts but only {len(sentences)} are available in the corpus."
        )
    return sentences[:target_count]


def generate_sentences(language: str, target_count: int) -> list[str]:
    return _generate_sentences_for_language(_normalize_language(language), target_count)
