from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SYMBOLS = [
    "_", ",", ".", "!", "?", "…", "~", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ",
    "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ", "ㅏ",
    "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅐ", "ㅔ", " ",
]


@dataclass
class TextTokenizer:
    symbols: list[str]

    def __post_init__(self) -> None:
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.pad_id = self.symbol_to_id["_"]

    def encode(self, text: str) -> list[int]:
        tokens = []
        for char in text:
            if char in self.symbol_to_id:
                tokens.append(self.symbol_to_id[char])
        if not tokens:
            tokens.append(self.pad_id)
        return tokens
