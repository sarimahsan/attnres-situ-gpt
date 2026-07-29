import torch
import pytest
from config.model_config import GPTConfig
from models.mlp import SiTUGLU, SwiGLU, MLP

def test_situ_glu_bounds():
    """Verify that pre-downprojection activations are strictly bounded by beta1 * beta2 = 100."""
    config = GPTConfig(variant='situ_glu', situ_beta1=4.0, situ_beta2=25.0)
    situ = SiTUGLU(config)
    
    # Test with extreme input values (+- 1000)
    x = torch.randn(8, 16, config.n_embd) * 100.0
    
    z_g = situ.w_gate(x)
    z_u = situ.w_up(x)
    
    gate = situ.beta1 * torch.tanh(z_g / situ.beta1) * torch.sigmoid(z_g)
    up = situ.beta2 * torch.tanh(z_u / situ.beta2)
    activation = gate * up
    
    # Absolute max activation value coordinate-wise must be <= 100.0
    max_act = activation.abs().max().item()
    assert max_act <= 100.0 + 1e-4, f"SiTU-GLU activation exceeded bound 100.0: got {max_act}"

def test_situ_glu_forward_backward():
    """Verify forward and backward pass through SiTU-GLU."""
    config = GPTConfig(variant='situ_glu')
    situ = SiTUGLU(config)
    x = torch.randn(2, 8, config.n_embd, requires_grad=True)
    
    out = situ(x)
    assert out.shape == (2, 8, config.n_embd)
    assert not torch.isnan(out).any(), "NaN detected in SiTU-GLU output"
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "NaN detected in SiTU-GLU input gradients"

def test_mlp_factory():
    """Verify MLP factory correctly instantiates SwiGLU vs SiTUGLU."""
    cfg_base = GPTConfig(variant='baseline')
    mlp_base = MLP(cfg_base)
    assert isinstance(mlp_base.net, SwiGLU)
    
    cfg_situ = GPTConfig(variant='situ_glu')
    mlp_situ = MLP(cfg_situ)
    assert isinstance(mlp_situ.net, SiTUGLU)
