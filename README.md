# Custom LLM

This repository provides a minimal yet complete example of how to build, train, and use a custom decoder-only transformer language model.

## Features

- Character-level tokenizer for quickly preparing small corpora.
- Configurable transformer implementation built with PyTorch.
- Training utilities that can infer the vocabulary size automatically.
- Helper function for autoregressive text generation.
- Automated tests that validate the forward pass and a short training loop.

## Getting started

Install the development dependencies (PyTorch is required for training):

```bash
pip install -e .[dev,train]
```

## Training the model

```python
from llm import ModelConfig, TrainingConfig, train_model, generate_text

text = "the quick brown fox jumps over the lazy dog. " * 10
model_config = ModelConfig(context_length=32, embedding_dim=128, num_layers=2, num_heads=4)
training_config = TrainingConfig(num_steps=100, batch_size=16)
model, tokenizer, losses = train_model(text, model_config, training_config)
print(generate_text(model, tokenizer, "the quick", max_new_tokens=20))
```

The vocabulary size is inferred automatically from the provided text when `vocab_size` is left at its default value of `0`.

## Running tests

```bash
pytest
```
