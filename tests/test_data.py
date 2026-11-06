
import pytest
import torch

from src.tokenizer import CharacterTokenizer
from src.data import (
    prepare_data,
    load_data,
    random_batch,
    evaluation_chunks,
)


def test_tokenizer_roundtrip_unknown_and_save(tmp_path):
    tok = CharacterTokenizer.fit("Ab\n!")

    assert tok.decode(tok.encode("Ab\n!")) == "Ab\n!"
    assert tok.encode("z") == [0]

    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = CharacterTokenizer.load(path)

    assert loaded.to_dict() == tok.to_dict()


def test_tokenizer_rejects_invalid_ids():
    tok = CharacterTokenizer.fit("abc")

    with pytest.raises(ValueError):
        tok.decode([-1])

    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])

    with pytest.raises(TypeError):
        tok.decode([1.5])

    with pytest.raises(TypeError):
        tok.encode(123)


def test_split_and_vocabulary_no_leakage(tmp_path):
    raw = tmp_path / "input.txt"
    text = "a" * 90 + "Z" * 10
    raw.write_bytes(text.encode("utf-8"))

    output = tmp_path / "processed"
    metadata = prepare_data(raw, output)

    data, tok = load_data(output)

    assert metadata["split_index"] == 90
    assert metadata["train_tokens"] == 90
    assert metadata["validation_tokens"] == 10

    assert len(data["train"]) == 90
    assert len(data["validation"]) == 10

    # Z only occurs in validation, so it must not enter vocabulary.
    assert "Z" not in tok.stoi
    assert (data["validation"] == tok.unk_id).all()

    assert metadata["vocab_size"] == tok.vocab_size


def test_raw_text_is_preserved(tmp_path):
    raw = tmp_path / "input.txt"

    text = "Ab\n!cd\r\nEF\n"
    raw.write_bytes(text.encode("utf-8"))

    output = tmp_path / "processed"
    prepare_data(raw, output)

    data, tok = load_data(output)

    split = int(0.9 * len(text))

    train_text = text[:split]
    validation_text = text[split:]

    assert tok.decode(data["train"].tolist()) == train_text

    # Known validation characters must be preserved.
    known_chars = "".join(
        c for c in validation_text if c in tok.stoi
    )

    decoded = tok.decode(data["validation"].tolist())
    assert decoded == "".join(
        c if c in tok.stoi else "<UNK>"
        for c in validation_text
    )


def test_random_batch_shape_and_shifting():
    ids = torch.arange(20, dtype=torch.long)

    x, y = random_batch(
        ids,
        batch_size=4,
        context_length=5,
        device="cpu"
    )

    assert x.shape == (4, 5)
    assert y.shape == (4, 5)

    # Every target is exactly the next token after its input.
    assert torch.equal(y, x + 1)


def test_random_batch_invalid_arguments():
    ids = torch.arange(10, dtype=torch.long)

    with pytest.raises(ValueError):
        random_batch(ids, 0, 4, "cpu")

    with pytest.raises(ValueError):
        random_batch(ids, 2, 0, "cpu")

    with pytest.raises(ValueError):
        random_batch(ids[:4], 2, 4, "cpu")

    with pytest.raises(ValueError):
        random_batch(ids.float(), 2, 4, "cpu")


def test_fixed_evaluation_chunks_no_missing_or_duplicate_targets():
    ids = torch.arange(9, dtype=torch.long)

    chunks = list(evaluation_chunks(ids, context_length=4))

    assert len(chunks) == 2

    # First chunk: targets 1, 2, 3, 4.
    assert torch.equal(
        chunks[0][0],
        torch.tensor([[0, 1, 2, 3]])
    )

    assert torch.equal(
        chunks[0][1],
        torch.tensor([[1, 2, 3, 4]])
    )

    # Second chunk: targets 5, 6, 7, 8.
    assert torch.equal(
        chunks[1][0],
        torch.tensor([[4, 5, 6, 7]])
    )

    assert torch.equal(
        chunks[1][1],
        torch.tensor([[5, 6, 7, 8]])
    )

    # Every target is evaluated exactly once.
    evaluated_targets = torch.cat(
        [y.flatten() for _, y in chunks]
    )

    assert torch.equal(evaluated_targets, ids[1:])


def test_fixed_evaluation_chunks_keep_final_partial_chunk():
    ids = torch.arange(12, dtype=torch.long)

    chunks = list(evaluation_chunks(ids, context_length=4))

    assert [y.shape[1] for _, y in chunks] == [4, 4, 3]

    evaluated_targets = torch.cat(
        [y.flatten() for _, y in chunks]
    )

    assert torch.equal(evaluated_targets, ids[1:])


def test_evaluation_max_tokens_counts_targets():
    ids = torch.arange(12, dtype=torch.long)

    chunks = list(
        evaluation_chunks(
            ids,
            context_length=4,
            max_tokens=5
        )
    )

    targets = torch.cat([y.flatten() for _, y in chunks])

    assert torch.equal(targets, torch.arange(1, 6))
    assert targets.numel() == 5


def test_evaluation_invalid_arguments():
    ids = torch.arange(10, dtype=torch.long)

    with pytest.raises(ValueError):
        list(evaluation_chunks(ids, 0))

    with pytest.raises(ValueError):
        list(evaluation_chunks(ids[:1], 4))

    with pytest.raises(ValueError):
        list(evaluation_chunks(ids, 4, max_tokens=0))


def test_load_data_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path)


def test_prepare_data_rejects_empty_corpus(tmp_path):
    raw = tmp_path / "empty.txt"
    raw.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        prepare_data(raw, tmp_path / "processed")
