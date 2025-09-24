"""Simple character-level tokenizer for demonstration purposes."""
from __future__ import annotations

from collections.abc import Iterable


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CharTokenizer:
    """Converts between strings and integer token ids."""

    def __init__(self, alphabet: Iterable[str] | None = None) -> None:
        base_vocab = [PAD_TOKEN, UNK_TOKEN]
        if alphabet is None:
            alphabet = []
        unique_chars = list(dict.fromkeys(alphabet))
        self.id_to_token = base_vocab + unique_chars
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    def fit(self, text: str) -> None:
        for char in text:
            if char not in self.token_to_id:
                self.token_to_id[char] = len(self.id_to_token)
                self.id_to_token.append(char)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(char, self.token_to_id[UNK_TOKEN]) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[idx] if 0 <= idx < len(self.id_to_token) else UNK_TOKEN for idx in ids)

    def pad_sequence(self, ids: list[int], length: int) -> list[int]:
        if len(ids) > length:
            return ids[-length:]
        return ids + [self.token_to_id[PAD_TOKEN]] * (length - len(ids))
