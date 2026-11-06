
import pytest
import torch

from src.model import LanguageModel


MODEL_CFG = {
    "n_layers": 2,
    "d_model": 128,
    "n_heads": 4,
    "d_ff": 512,
    "context_length": 128,
    "dropout": 0.1,
}


def test_model_rejects_invalid_vocab_size():
    with pytest.raises(ValueError):
        LanguageModel(0, MODEL_CFG)


def test_model_rejects_invalid_config():
    invalid_cfg = dict(MODEL_CFG)
    invalid_cfg["n_heads"] = 0

    with pytest.raises(ValueError):
        LanguageModel(66, invalid_cfg)


def test_model_rejects_invalid_token_ids():
    model = LanguageModel(66, MODEL_CFG)

    with pytest.raises(TypeError):
        model(torch.ones(2, 8, dtype=torch.float32))

    with pytest.raises(ValueError):
        model(torch.full((2, 8), -1, dtype=torch.long))

    with pytest.raises(ValueError):
        model(torch.full((2, 8), 66, dtype=torch.long))

    with pytest.raises(ValueError):
        model(torch.zeros(2, 129, dtype=torch.long))


def test_model_weight_tying():
    model = LanguageModel(66, MODEL_CFG)

    assert (
        model.token_embedding.weight
        is model.head.weight
    )


def test_model_output_shape_and_parameter_count():
    model = LanguageModel(66, MODEL_CFG)

    ids = torch.randint(0, 66, (8, 32))

    logits = model(ids)

    assert logits.shape == (8, 32, 66)

    assert (
        model.parameter_count()
        == model.expected_parameter_count()
    )


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(42)
    model = LanguageModel(66, MODEL_CFG).eval()

    ids = torch.randint(0, 66, (2, 16))
    changed = ids.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 66

    with torch.no_grad():
        original_logits = model(ids)
        changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, :8],
        changed_logits[:, :8],
        rtol=1e-5,
        atol=1e-6,
    )


def test_model_backward_reaches_all_parameters():
    torch.manual_seed(42)
    model = LanguageModel(66, MODEL_CFG)
    ids = torch.randint(0, 66, (2, 17))

    # Dùng token kế tiếp làm target.
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 66),
        targets.reshape(-1),
    )

    assert torch.isfinite(loss).item()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all().item(), (
            f"Non-finite gradient: {name}"
        )


def test_model_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = LanguageModel(66, MODEL_CFG).eval()
    ids = torch.randint(0, 66, (2, 16))

    with torch.no_grad():
        expected = model(ids)

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    restored = LanguageModel(66, MODEL_CFG)
    restored.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    restored.eval()

    assert restored.head.weight is restored.token_embedding.weight

    with torch.no_grad():
        actual = restored(ids)

    torch.testing.assert_close(actual, expected)