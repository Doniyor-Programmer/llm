"""Training utilities for the custom language model."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .dataset import TextDataset, batch_to_device
from .model import CustomLLM
from .tokenizer import CharTokenizer


def train_model(
    text: str,
    model_config: ModelConfig,
    training_config: TrainingConfig | None = None,
) -> tuple[CustomLLM, CharTokenizer, list[float]]:
    """Train a :class:`CustomLLM` on the provided text corpus."""

    if training_config is None:
        training_config = TrainingConfig()
    training_config.validate()

    tokenizer = CharTokenizer()
    dataset = TextDataset(text, tokenizer, model_config.context_length)
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)
    if model_config.vocab_size == 0:
        model_config = replace(model_config, vocab_size=tokenizer.vocab_size)
    elif model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "ModelConfig.vocab_size does not match tokenizer vocabulary size; "
            "set vocab_size to 0 to infer automatically."
        )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if len(dataloader) == 0:
        raise ValueError("Training dataset is empty; adjust batch size or provide more text")

    device = torch.device(training_config.device)
    model = CustomLLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    model.train()
    data_iter: Iterable[list[tuple[torch.Tensor, torch.Tensor]]] = iter(dataloader)
    for step in range(1, training_config.num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        inputs, targets = batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, model_config.vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % training_config.log_interval == 0:
            print(f"step={step} loss={loss.item():.4f}")

    return model, tokenizer, losses


def generate_text(
    model: CustomLLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> str:
    """Generate text from a trained model using the provided prompt."""

    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(generated_ids[0].tolist())


if __name__ == "__main__":  # pragma: no cover - example usage
    sample_text = "hello world! this is a demo dataset for the custom llm."
    model_cfg = ModelConfig(context_length=16)
    training_cfg = TrainingConfig(num_steps=10, batch_size=4, log_interval=5)
    model, tokenizer, _ = train_model(sample_text, model_cfg, training_cfg)
    print(generate_text(model, tokenizer, "hello", max_new_tokens=20))
