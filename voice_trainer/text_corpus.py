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


def _normalize_prompt_languages(
    prompt_languages: list[dict] | None,
    fallback_language: str,
) -> list[dict]:
    if not prompt_languages:
        return [{"language": _normalize_language(fallback_language), "weight": 1.0}]

    normalized = []
    for item in prompt_languages:
        if not isinstance(item, dict):
            raise ValueError("Each prompt language entry must be an object.")
        language = item.get("language")
        weight = item.get("weight", 1.0)
        if not isinstance(language, str):
            raise ValueError("Each prompt language entry must include a string language.")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("Each prompt language entry must include a positive numeric weight.")
        normalized.append(
            {
                "language": _normalize_language(language),
                "weight": float(weight),
            }
        )
    return normalized


def _allocate_prompt_counts(language_specs: list[dict], target_count: int) -> list[dict]:
    total_weight = sum(spec["weight"] for spec in language_specs)
    allocations = []
    assigned = 0
    for spec in language_specs:
        exact = (target_count * spec["weight"]) / total_weight
        count = int(exact)
        allocations.append({**spec, "count": count, "remainder": exact - count})
        assigned += count

    for spec in sorted(allocations, key=lambda item: item["remainder"], reverse=True)[: target_count - assigned]:
        spec["count"] += 1

    return allocations


def generate_prompt_items(
    *,
    fallback_language: str,
    target_count: int,
    prompt_languages: list[dict] | None = None,
) -> list[dict]:
    language_specs = _allocate_prompt_counts(
        _normalize_prompt_languages(prompt_languages, fallback_language),
        target_count,
    )
    buckets: list[list[dict]] = []
    for spec in language_specs:
        texts = _generate_sentences_for_language(spec["language"], spec["count"])
        buckets.append(
            [{"text": text, "language": spec["language"]} for text in texts]
        )

    prompts: list[dict] = []
    active_indices = [index for index, bucket in enumerate(buckets) if bucket]
    while active_indices:
        next_indices = []
        for index in active_indices:
            bucket = buckets[index]
            if bucket:
                prompts.append(bucket.pop(0))
            if bucket:
                next_indices.append(index)
        active_indices = next_indices

    if len(prompts) != target_count:
        raise ValueError(f"Prompt generation produced {len(prompts)} items for target count {target_count}.")
    return prompts
