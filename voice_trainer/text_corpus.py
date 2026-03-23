from __future__ import annotations

from collections import OrderedDict


OPENINGS = [
    "오늘은",
    "방금",
    "조금 전에",
    "아까",
    "이번 주에는",
    "요즘은",
    "퇴근하고 나서",
    "학교 끝나고",
    "주말마다",
    "집에 가는 길에",
]

SUBJECTS = [
    "나는",
    "우리는",
    "동생은",
    "친구는",
    "엄마는",
    "아빠는",
    "선생님은",
    "팀원은",
    "옆집 사람은",
    "매니저는",
]

PLACES = [
    "집에서",
    "학교에서",
    "도서관에서",
    "카페에서",
    "회사에서",
    "연습실에서",
    "버스 안에서",
    "지하철역에서",
    "공원에서",
    "편의점 앞에서",
]

VERBS = [
    "천천히 정리하고 있어",
    "다시 확인해 보고 있어",
    "잠깐 쉬고 있어",
    "메모를 남기고 있어",
    "전화할 준비를 하고 있어",
    "계획을 다시 세우고 있어",
    "자료를 찾아보고 있어",
    "조용히 연습하고 있어",
    "일정을 맞추고 있어",
    "필요한 걸 챙기고 있어",
]

ENDINGS = [
    "그래서 너무 급할 건 없어.",
    "생각보다 금방 끝날 것 같아.",
    "조금만 더 하면 마무리될 거야.",
    "지금 상태로도 크게 문제는 없어.",
    "마음만 급하지 실제로는 괜찮아.",
    "오늘 안에는 충분히 끝낼 수 있어.",
]


def generate_korean_sentences(target_count: int) -> list[str]:
    sentences = OrderedDict()
    for opening in OPENINGS:
        for subject in SUBJECTS:
            for place in PLACES:
                for verb in VERBS:
                    for ending in ENDINGS:
                        sentence = f"{opening} {subject} {place} {verb}. {ending}"
                        sentence = " ".join(sentence.split())
                        sentences[sentence] = None
                        if len(sentences) >= target_count:
                            return list(sentences.keys())
    return list(sentences.keys())


def generate_sentences(language: str, target_count: int) -> list[str]:
    normalized = language.strip().lower()
    if normalized == "korean":
        return generate_korean_sentences(target_count)
    raise ValueError(f"Unsupported generation language: {language}")
