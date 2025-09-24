"""Implementation of a lightweight transformer based language model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .config import ModelConfig


class TransformerBlock(nn.Module):
    """A single decoder-only transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.embedding_dim),
        )
        self.ln2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.ln2(x)
        return x


class CustomLLM(nn.Module):
    """A custom GPT-style decoder-only language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 1D or 2D tensor")

        batch_size, seq_length = input_ids.shape
        if seq_length > self.config.context_length:
            raise ValueError("Sequence length exceeds model context length")

        device = input_ids.device
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.dropout(x)

        attn_mask = self._build_causal_mask(seq_length, device=device)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def _build_causal_mask(self, seq_length: int, device: torch.device) -> Tensor:
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.context_length :]
            logits = self(context)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
