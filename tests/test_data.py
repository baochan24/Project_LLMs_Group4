import torch
from src.tokenizer import CharacterTokenizer
from src.data import prepare_data, load_data, evaluation_chunks


def test_tokenizer_roundtrip_unknown_and_save(tmp_path):
    tok = CharacterTokenizer.fit('Ab\n!')
    assert tok.decode(tok.encode('Ab\n!')) == 'Ab\n!'
    assert tok.encode('z') == [0]
    path = tmp_path / 'tokenizer.json'
    tok.save(path)
    assert CharacterTokenizer.load(path).to_dict() == tok.to_dict()


def test_split_and_shifting(tmp_path):
    raw = tmp_path / 'input.txt'
    text = 'a' * 90 + 'Z' * 10
    raw.write_bytes(text.encode())
    out = tmp_path / 'processed'
    meta = prepare_data(raw, out)
    data, tok = load_data(out)
    assert meta['split_index'] == 90
    assert len(data['train']) == 90 and len(data['validation']) == 10
    assert 'Z' not in tok.stoi
    assert (data['validation'] == tok.unk_id).all()
    ids = torch.arange(12)
    chunks = list(evaluation_chunks(ids, 4))
    assert torch.equal(torch.cat([y.flatten() for _, y in chunks]), ids[1:])
    for x, y in chunks:
        assert torch.equal(y, x + 1)
