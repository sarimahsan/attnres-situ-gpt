import torch
import pytest
from config.model_config import GPTConfig
from models.attn_res import AttnResRouter, LayerRMSNorm

def test_attn_res_router_softmax_and_shapes():
    """Verify AttnRes router produces valid softmax probabilities and correct output shapes."""
    config = GPTConfig(variant='attn_res', n_embd=512)
    router = AttnResRouter(layer_idx=3, config=config)
    
    # 4 prior cached layer states (l=4: layer_idx=3)
    B, T, d = 2, 16, 512
    cached_values = [torch.randn(B, T, d) for _ in range(4)]
    cached_keys = [torch.randn(B, T, d) for _ in range(4)]
    
    x_l, alpha = router(cached_values, cached_keys)
    
    # Shape checks
    assert x_l.shape == (B, T, d)
    assert alpha.shape == (4, B, T)
    
    # Softmax condition check: sum over layer dimension (dim=0) must be 1.0
    alpha_sum = alpha.sum(dim=0)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5), \
        "AttnRes depth softmax does not sum to 1.0"

def test_attn_res_router_backward():
    """Verify gradient flow through AttnRes pseudo-queries and cached representations."""
    config = GPTConfig(variant='attn_res', n_embd=512)
    router = AttnResRouter(layer_idx=1, config=config)
    
    B, T, d = 2, 8, 512
    v0 = torch.randn(B, T, d, requires_grad=True)
    v1 = torch.randn(B, T, d, requires_grad=True)
    
    cached_values = [v0, v1]
    cached_keys = [v0, v1]
    
    x_l, alpha = router(cached_values, cached_keys)
    loss = x_l.sum()
    loss.backward()
    
    assert router.q_l.grad is not None, "Gradients failed to reach AttnRes pseudo-query q_l"
    assert v0.grad is not None and v1.grad is not None, "Gradients failed to flow back to cached layer history"
