import torch
import pytest
from config.model_config import GPTConfig
from models.gpt import GPT

@pytest.mark.parametrize("variant", ["baseline", "attn_res", "situ_glu", "both"])
def test_gpt_variant_instantiation_and_forward(variant: str):
    """Verify that all 4 model variants build, run forward/backward without NaN, and produce logits."""
    config = GPTConfig(
        variant=variant,
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=128,
        vocab_size=1000
    )
    model = GPT(config)
    
    # Dummy input batch
    idx = torch.randint(0, 1000, (2, 64))
    targets = torch.randint(0, 1000, (2, 64))
    
    logits, loss = model(idx, targets)
    
    assert logits.shape == (2, 64, 1000)
    assert loss is not None
    assert not torch.isnan(loss), f"NaN loss detected for variant {variant}"
    
    loss.backward()
    
    # Verify parameter gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name} in {variant}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name} for {variant}"

def test_parameter_count_matching():
    """Verify exact parameter matching across baseline, +AttnRes, +SiTU, +Both."""
    base_cfg = GPTConfig(variant="baseline", n_layer=6, n_head=8, n_embd=512)
    m_base = GPT(base_cfg)
    
    attn_cfg = GPTConfig(variant="attn_res", n_layer=6, n_head=8, n_embd=512)
    m_attn = GPT(attn_cfg)
    
    situ_cfg = GPTConfig(variant="situ_glu", n_layer=6, n_head=8, n_embd=512)
    m_situ = GPT(situ_cfg)
    
    both_cfg = GPTConfig(variant="both", n_layer=6, n_head=8, n_embd=512)
    m_both = GPT(both_cfg)
    
    p_base = m_base.get_num_params()
    p_attn = m_attn.get_num_params()
    p_situ = m_situ.get_num_params()
    p_both = m_both.get_num_params()
    
    print(f"Params Baseline: {p_base:,} | AttnRes: {p_attn:,} | SiTU: {p_situ:,} | Both: {p_both:,}")
    
    # SiTU-GLU uses same linear projection dimensions as SwiGLU -> exact equal params
    assert p_base == p_situ, f"Baseline ({p_base}) and SiTU ({p_situ}) param counts must match exactly"
    assert p_attn == p_both, f"AttnRes ({p_attn}) and Both ({p_both}) param counts must match exactly"
    
    # AttnRes adds L * d_model pseudo-query params + 1 * d_model LayerRMSNorm scale param ((6 + 1) * 512 = 3584 params) -> (< 0.02% overhead)
    diff = p_attn - p_base
    expected_diff = (6 + 1) * 512
    assert diff == expected_diff, f"AttnRes parameter difference should be exactly (L+1)*d ({expected_diff}), got {diff}"


def test_flop_estimation():
    """Verify FLOP estimation per token."""
    config = GPTConfig(variant="baseline", n_layer=6, n_head=8, n_embd=512, block_size=512)
    model = GPT(config)
    flops = model.estimate_flops_per_token()
    assert flops > 0, "FLOP estimation must be positive"


@pytest.mark.parametrize("variant", ["baseline", "attn_res", "situ_glu", "both"])
def test_configure_optimizers(variant: str):
    """Verify that configure_optimizers assigns all parameters to decay/no_decay without errors."""
    config = GPTConfig(variant=variant, n_layer=4, n_head=4, n_embd=256)
    model = GPT(config)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type="cpu")
    assert optimizer is not None
    assert len(optimizer.param_groups) == 2
