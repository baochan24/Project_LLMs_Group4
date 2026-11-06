
"""Tests for training, optimizer, checkpoint and resume."""

import copy
import json

import pytest
import torch

from src.data import prepare_data
from src.model import LanguageModel
from src.tokenizer import CharacterTokenizer
from src.train import (
    make_optimizer,
    save_checkpoint,
    load_checkpoint,
    train,
)


def make_model_config():
    """Small model configuration for fast CPU tests."""
    return {
        "n_layers": 1,
        "d_model": 8,
        "n_heads": 2,
        "d_ff": 16,
        "context_length": 4,
        "dropout": 0.1,
    }


def make_training_config(tmp_path, name, data_dir):
    """Create an isolated training configuration."""
    return {
        "seed": 42,
        "device": "cpu",
        "data_dir": str(data_dir),
        "output_dir": str(tmp_path / name / "outputs"),
        "checkpoint_dir": str(tmp_path / name / "checkpoints"),
        "model": make_model_config(),
        "training": {
            "batch_size": 2,
            "steps": 4,
            "eval_interval": 2,
            "eval_tokens": 24,
            "num_threads": 1,
            "learning_rate": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
        },
        "generation": {
            "prompt": "ab",
            "max_new_tokens": 2,
        },
    }


def assert_nested_equal(actual, expected):
    """Compare nested dictionaries, lists and tensors."""
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
        )
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a deterministic local dataset."""
    raw_path = tmp_path / "input.txt"
    text = "abcdefghij" * 30
    raw_path.write_text(text, encoding="utf-8")

    data_dir = tmp_path / "processed"
    prepare_data(raw_path, data_dir)

    return data_dir


def test_optimizer_has_no_duplicate_parameters():
    """Tied embedding/head weights must appear only once."""
    model = LanguageModel(10, make_model_config())
    optimizer = make_optimizer(
        model,
        {
            "learning_rate": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    )

    parameter_ids = []

    for group in optimizer.param_groups:
        parameter_ids.extend(
            id(parameter)
            for parameter in group["params"]
        )

    assert len(parameter_ids) == len(set(parameter_ids))

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    assert set(parameter_ids) == model_parameter_ids


def test_checkpoint_roundtrip_restores_training_state(tmp_path):
    """Save and restore model, optimizer, tokenizer and RNG."""
    torch.manual_seed(42)

    cfg = {
        "model": make_model_config(),
        "training": {"num_threads": 1},
    }

    tokenizer = CharacterTokenizer.fit("abcdefghij")
    model = LanguageModel(tokenizer.vocab_size, cfg["model"])

    optimizer = make_optimizer(
        model,
        {
            "learning_rate": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    )

    # Create optimizer state through one update.
    ids = torch.randint(0, tokenizer.vocab_size, (2, 4))
    logits = model(ids)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, tokenizer.vocab_size),
        ids.reshape(-1),
    )

    loss.backward()
    optimizer.step()

    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    torch.manual_seed(123)
    expected_rng = torch.get_rng_state().clone()

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        tokenizer,
        cfg,
        step=5,
        best=1.25,
        data_sha256="test_dataset_hash",
    )

    # Change RNG and create a new model through checkpoint loading.
    torch.manual_seed(999)

    restored_model, restored_tokenizer, payload = load_checkpoint(
        path,
        device="cpu",
    )

    assert_nested_equal(
        restored_model.state_dict(),
        expected_model,
    )

    assert restored_tokenizer.to_dict() == tokenizer.to_dict()

    assert payload["step"] == 5
    assert payload["best_validation_loss"] == 1.25
    assert payload["data_sha256"] == "test_dataset_hash"

    restored_optimizer = make_optimizer(
        restored_model,
        {
            "learning_rate": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    )

    restored_optimizer.load_state_dict(payload["optimizer"])

    assert_nested_equal(
        restored_optimizer.state_dict(),
        expected_optimizer,
    )

    torch.testing.assert_close(
      payload["torch_rng"],
      expected_rng,
      rtol=0,
      atol=0,
    )


def test_continuous_training_matches_resumed_training(
    tmp_path,
    tiny_dataset,
):
    """Continuous 4 steps must match 2 steps + resume 2 steps."""
    continuous_cfg = make_training_config(
        tmp_path,
        "continuous",
        tiny_dataset,
    )

    resumed_cfg = make_training_config(
        tmp_path,
        "resumed",
        tiny_dataset,
    )

    # Scenario A: train continuously for 4 steps.
    train(
        continuous_cfg,
        steps=4,
    )

    # Scenario B: train for 2 steps.
    train(
        resumed_cfg,
        steps=2,
    )

    checkpoint_path = (
        tmp_path
        / "resumed"
        / "checkpoints"
        / "latest.pt"
    )

    assert checkpoint_path.is_file()

    # Resume from step 2 and continue to total step 4.
    train(
        resumed_cfg,
        resume=str(checkpoint_path),
        steps=4,
    )

    continuous_checkpoint = load_checkpoint(
        tmp_path / "continuous" / "checkpoints" / "latest.pt"
    )

    resumed_checkpoint = load_checkpoint(
        tmp_path / "resumed" / "checkpoints" / "latest.pt"
    )

    continuous_model, _, continuous_payload = continuous_checkpoint
    resumed_model, _, resumed_payload = resumed_checkpoint

    assert continuous_payload["step"] == 4
    assert resumed_payload["step"] == 4

    # Compare final model parameters.
    assert_nested_equal(
        continuous_model.state_dict(),
        resumed_model.state_dict(),
    )

    # Compare optimizer states.
    assert_nested_equal(
        continuous_payload["optimizer"],
        resumed_payload["optimizer"],
    )

    # Compare final random generator states.
    assert_nested_equal(
        continuous_payload["torch_rng"],
        resumed_payload["torch_rng"],
    )


def test_resume_rejects_mismatched_config(tmp_path, tiny_dataset):
    """Resume must reject a different training configuration."""
    cfg = make_training_config(
        tmp_path,
        "mismatch",
        tiny_dataset,
    )

    train(cfg, steps=2)

    checkpoint_path = (
        tmp_path
        / "mismatch"
        / "checkpoints"
        / "latest.pt"
    )

    changed_cfg = copy.deepcopy(cfg)
    changed_cfg["training"]["learning_rate"] = 0.001

    with pytest.raises(ValueError, match="mismatch"):
        train(
            changed_cfg,
            resume=str(checkpoint_path),
            steps=4,
        )

def test_resume_restores_best_checkpoint_from_its_own_history(
    tmp_path,
    tiny_dataset,
):
    """A future best.pt must not survive a rollback to an older latest.pt."""

    cfg = make_training_config(
        tmp_path,
        "best_history",
        tiny_dataset,
    )

    latest_path = (
        tmp_path
        / "best_history"
        / "checkpoints"
        / "latest.pt"
    )

    best_path = (
        tmp_path
        / "best_history"
        / "checkpoints"
        / "best.pt"
    )

    # Save the legitimate history at step 2.
    train(cfg, steps=2)
    historical_latest = latest_path.read_bytes()

    historical_payload = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=True,
    )

    assert historical_payload["step"] == 2
    assert historical_payload["best_checkpoint"] is not None

    historical_best_step = (
        historical_payload["best_checkpoint"]["step"]
    )

    # Create a newer checkpoint on disk.
    train(cfg, steps=4)

    future_payload = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=True,
    )

    assert future_payload["step"] == 4

    # Simulate a stale best.pt from a future run.
    torch.save(future_payload, best_path)

    # Roll back latest.pt to the historical step 2.
    latest_path.write_bytes(historical_latest)

    # Resume from step 2 and continue to step 3.
    train(
        cfg,
        resume=str(latest_path),
        steps=3,
    )

    restored_best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    restored_latest = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=True,
    )

    assert restored_latest["step"] == 3

    # The best checkpoint must belong to the resumed history.
    assert restored_best["step"] <= 3

    # best.pt and latest.pt must agree on the best snapshot.
    assert (
        restored_best["step"]
        == restored_latest["best_checkpoint"]["step"]
    )

    assert (
        restored_best["best_validation_loss"]
        == restored_latest["best_checkpoint"]["best_validation_loss"]
    )

    assert historical_best_step <= 2
