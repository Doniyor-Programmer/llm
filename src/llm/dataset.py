"""Datasets and utilities for training the custom LLM."""
from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Creates input/target sequences from a corpus of text."""

    def __init__(self, text: str, tokenizer: CharTokenizer, context_length: int) -> None:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        self.tokenizer = tokenizer
        self.tokenizer.fit(text)
        self.context_length = context_length
        self.tokens = self.tokenizer.encode(text)
        if len(self.tokens) < context_length + 1:
            raise ValueError("Text is too short for the chosen context length")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        input_tokens = self.tokens[idx : idx + self.context_length]
        target_tokens = self.tokens[idx + 1 : idx + self.context_length + 1]
        x = torch.tensor(input_tokens, dtype=torch.long)
        y = torch.tensor(target_tokens, dtype=torch.long)
        return x, y


def batch_to_device(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        inputs_tensor, targets_tensor = batch
    else:
        inputs, targets = zip(*batch)  # type: ignore[arg-type]
        inputs_tensor = torch.stack(inputs)
        targets_tensor = torch.stack(targets)
    return inputs_tensor.to(device), targets_tensor.to(device)
