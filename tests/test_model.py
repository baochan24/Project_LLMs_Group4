import copy
import json
import torch
from torch.nn import functional as F
from src.model import LanguageModel
from src.attention import CausalSelfAttention
from src.train import make_optimizer, save_checkpoint, load_checkpoint, train
from src.tokenizer import CharacterTokenizer
from src.generate import generate, filter_logits
from src.evaluate import evaluate
from src.data import prepare_data

CFG = {'d_model': 16, 'n_heads': 4, 'n_layers': 2, 'context_length': 8, 'd_ff': 64, 'dropout': 0.1}


def test_attention_reference():
    torch.manual_seed(7)
    attn = CausalSelfAttention(16, 4, 8, 0).eval()
    x = torch.randn(2, 5, 16)
    q, k, v = [z.reshape(2, 5, 4, 4).transpose(1, 2) for z in attn.qkv(x).chunk(3, -1)]
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).contiguous().view(2, 5, 16))
    torch.testing.assert_close(attn(x), ref, atol=1e-6, rtol=1e-5)


def test_shape_causality_count_backward_checkpoint(tmp_path):
    torch.manual_seed(42)
    model = LanguageModel(5, CFG).eval()
    assert model.parameter_count() == model.expected_parameter_count()
    assert model.head.weight is model.token_embedding.weight
    x = torch.randint(5, (2, 8))
    y = x.clone()
    y[:, 4:] = (y[:, 4:] + 1) % 5
    assert model(x).shape == (2, 8, 5)
    torch.testing.assert_close(model(x)[:, :4], model(y)[:, :4], atol=1e-6, rtol=1e-5)
    t = {'learning_rate': 3e-4, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-8}
    opt = make_optimizer(model, t)
    params = [p for group in opt.param_groups for p in group['params']]
    assert len(params) == len(set(map(id, params)))
    old = model.token_embedding.weight.detach().clone()
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    assert not torch.equal(old, model.token_embedding.weight)
    tok = CharacterTokenizer.fit('abcd')
    path = tmp_path / 'test.pt'
    save_checkpoint(path, model, opt, tok, {'model': CFG}, 1, 2.0, 'test')
    loaded, _, _ = load_checkpoint(path)
    loaded.eval()
    torch.testing.assert_close(model(x), loaded(x), atol=0, rtol=0)


def test_evaluation_weighted_and_generation():
    tok = CharacterTokenizer.fit('abc')
    model = LanguageModel(tok.vocab_size, CFG)
    ids = torch.tensor(tok.encode('abcabcabcabc'))
    result = evaluate(model, ids, 'cpu')
    assert result['tokens'] == len(ids) - 1
    assert result['loss'] > 0
    one = generate(model, tok, 'abc', max_new_tokens=12, top_p=0.9, seed=7)
    two = generate(model, tok, 'abc', max_new_tokens=12, top_p=0.9, seed=7)
    assert one == two and len(one['generated_ids']) == 12
    assert model.training  # eval/generation restore previous mode
    filtered = filter_logits(torch.tensor([[9., 3., 2., 1.]]), top_k=1)
    assert filtered.softmax(-1).tolist() == [[1., 0., 0., 0.]]


def test_end_to_end_resume(tmp_path):
    cfg = json.loads(open('configs/baseline.json').read())
    cfg['model'] = copy.deepcopy(CFG)
    cfg['data_dir'] = str(tmp_path / 'data')
    cfg['output_dir'] = str(tmp_path / 'outputs')
    cfg['checkpoint_dir'] = str(tmp_path / 'checkpoints')
    cfg['training'].update(steps=3, eval_interval=1, eval_tokens=32)
    cfg['generation']['prompt'] = 'abc'
    raw = tmp_path / 'input.txt'
    raw.write_text('abc\n' * 100)
    prepare_data(raw, cfg['data_dir'])
    train(cfg, steps=2)
    path = tmp_path / 'checkpoints/latest.pt'
    train(cfg, resume=path, steps=3)
    resumed, tok, state = load_checkpoint(path)
    assert state['step'] == 3
    # Same uninterrupted CPU run should agree exactly on this fixed test.
    cfg['output_dir'] = str(tmp_path / 'full_outputs')
    cfg['checkpoint_dir'] = str(tmp_path / 'full_checkpoints')
    train(cfg, steps=3)
    full, _, _ = load_checkpoint(tmp_path / 'full_checkpoints/latest.pt')
    for p, q in zip(resumed.parameters(), full.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert len(generate(resumed, tok, 'abc', max_new_tokens=2)['generated_ids']) == 2
