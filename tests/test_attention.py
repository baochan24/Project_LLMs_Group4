
import pytest
import torch

from src.attention import CausalSelfAttention


@pytest.mark.parametrize(
    "d_model,n_heads,context_length,dropout",
    [
        (0, 4, 16, 0.1),
        (128, 0, 16, 0.1),
        (128, 4, 0, 0.1),
        (127, 4, 16, 0.1),
        (128, 4, 16, 1.0),
        (128, 4, 16, -0.1),
    ],
)
def test_attention_rejects_invalid_config(
    d_model, n_heads, context_length, dropout
):
    with pytest.raises(ValueError):
        CausalSelfAttention(
            d_model, n_heads, context_length, dropout
        )


def test_attention_rejects_invalid_input():
    model = CausalSelfAttention(128, 4, 16, 0.1)

    with pytest.raises(ValueError):
        model(torch.randn(2, 16))

    with pytest.raises(ValueError):
        model(torch.randn(2, 17, 128))

    with pytest.raises(ValueError):
        model(torch.randn(2, 16, 64))

    with pytest.raises(TypeError):
        model(torch.ones(2, 16, 128, dtype=torch.long))

def test_attention_weights_are_causal():
    torch.manual_seed(42)
    model = CausalSelfAttention(32, 4, 16, 0.0).eval()
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        y, weights = model(x, return_weights=True)

    assert y.shape == (2, 8, 32)
    assert weights.shape == (2, 4, 8, 8)
    assert torch.isfinite(y).all()
    assert torch.count_nonzero(weights.triu(diagonal=1)).item() == 0

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
    )


def test_attention_matches_pytorch_reference():
    torch.manual_seed(42)
    model = CausalSelfAttention(32, 4, 16, 0.0).eval()
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        actual = model(x)

        q, k, v = model.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            z.reshape(2, 8, 4, 8).transpose(1, 2)
            for z in (q, k, v)
        ]

        reference = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True,
        )
        reference = reference.transpose(1, 2).contiguous().view(2, 8, 32)
        reference = model.proj(reference)

    torch.testing.assert_close(
        actual, reference, rtol=1e-4, atol=1e-6
    )