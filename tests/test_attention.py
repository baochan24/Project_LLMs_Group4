import torch
import torch.nn.functional as F

from src.attention import CausalSelfAttention

def test_attention_output_shape():
    attention = CausalSelfAttention(
        d_model=16, n_heads=4, context_length=8, dropout=0.0
    )
    x = torch.randn(2, 5, 16)

    y = attention(x)

    assert y.shape == (2, 5, 16)

def test_attention_weights_are_causal():
    attention = CausalSelfAttention(
        d_model=16, n_heads=4, context_length=8, dropout=0.0
    ).eval()
    x = torch.randn(2, 5, 16)

    _, weights = attention(x, return_weights=True)

    future_weights = torch.triu(weights, diagonal=1)
    torch.testing.assert_close(
        future_weights,
        torch.zeros_like(future_weights),
        atol=0.0,
        rtol=0.0,
    )

def test_future_tokens_do_not_change_past_outputs():
    torch.manual_seed(11)
    attention = CausalSelfAttention(
        d_model=16, n_heads=4, context_length=8, dropout=0.0
    ).eval()

    x1 = torch.randn(2, 6, 16)
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])

    y1 = attention(x1)
    y2 = attention(x2)

    torch.testing.assert_close(y1[:, :4, :], y2[:, :4, :], atol=1e-6, rtol=1e-5)

def test_attention_matches_pytorch_reference():
    torch.manual_seed(7)
    attention = CausalSelfAttention(
        d_model=16, n_heads=4, context_length=8, dropout=0.0
    ).eval()
    x = torch.randn(2, 5, 16)

    q, k, v = [
        part.reshape(2, 5, 4, 4).transpose(1, 2)
        for part in attention.qkv(x).chunk(3, dim=-1)
    ]
    reference = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True
    )
    reference = attention.proj(
        reference.transpose(1, 2).contiguous().view(2, 5, 16)
    )

    torch.testing.assert_close(
        attention(x), reference, atol=1e-6, rtol=1e-5
    )