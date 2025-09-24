import torch

from llm import (
    CharTokenizer,
    CustomLLM,
    ModelConfig,
    TrainingConfig,
    generate_text,
    train_model,
)
from llm.dataset import TextDataset


def build_dataset(text: str, context_length: int) -> tuple[TextDataset, CharTokenizer]:
    tokenizer = CharTokenizer()
    dataset = TextDataset(text, tokenizer, context_length=context_length)
    return dataset, tokenizer


def test_model_forward_pass_produces_expected_shape() -> None:
    text = "hello world!" * 3
    context_length = 8
    dataset, tokenizer = build_dataset(text, context_length)
    config = ModelConfig(
        context_length=context_length,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        vocab_size=tokenizer.vocab_size,
    )
    model = CustomLLM(config)
    inputs = torch.stack([dataset[i][0] for i in range(2)])
    logits = model(inputs)
    assert logits.shape == (2, context_length, tokenizer.vocab_size)


def test_training_loop_and_generation() -> None:
    text = "the quick brown fox jumps over the lazy dog. " * 5
    context_length = 12
    model_config = ModelConfig(
        context_length=context_length,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    training_config = TrainingConfig(num_steps=3, batch_size=4, log_interval=10)
    model, tokenizer, losses = train_model(text, model_config, training_config)

    assert isinstance(model, CustomLLM)
    assert tokenizer.vocab_size > 0
    assert len(losses) == training_config.num_steps
    assert torch.isfinite(torch.tensor(losses)).all()

    generated = generate_text(model, tokenizer, "the quick", max_new_tokens=5)
    assert generated.startswith("the quick")
    assert len(generated) >= len("the quick")
