"""Configuration objects for the custom language model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration parameters describing the transformer architecture."""

    context_length: int
    vocab_size: int = 0
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    feedforward_dim: int | None = None

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.feedforward_dim is None:
            self.feedforward_dim = 4 * self.embedding_dim
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")


@dataclass
class TrainingConfig:
    """Configuration parameters for the training loop."""

    batch_size: int = 16
    learning_rate: float = 3e-4
    num_steps: int = 200
    device: str = "cpu"
    log_interval: int = 50

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
