import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import GPTConfig

class LayerRMSNorm(nn.Module):
    """Simple RMSNorm used for layer history normalization in AttnRes."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.scale


class AttnResRouter(nn.Module):
    """
    Full Attention Residuals (AttnRes) router for a single layer l.
    
    Routes inputs across all prior layer outputs [y_0, y_1, ..., y_{l-1}].
    Uses cached normalized keys v_i = RMSNorm(y_i) to avoid O(L^2) key recomputations.
    """
    def __init__(self, layer_idx: int, config: GPTConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.n_embd
        
        # Learnable pseudo-query vector q_l for layer l
        self.q_l = nn.Parameter(torch.randn(config.n_embd) / math.sqrt(config.n_embd))
        self.scale = 1.0 / math.sqrt(config.n_embd)

    def forward(self, cached_values: list[torch.Tensor], cached_keys: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cached_values: list of [y_0, ..., y_{l-1}], each shape (B, T, d)
        cached_keys: list of [v_0, ..., v_{l-1}], each shape (B, T, d) where v_i = RMSNorm(y_i)
        
        Returns:
            aggregated input x_l of shape (B, T, d)
            softmax alpha weights of shape (l, B, T)
        """
        l = len(cached_keys)
        assert l == self.layer_idx + 1, f"Expected {self.layer_idx + 1} cached states, got {l}"
        
        # Stack values and keys along layer dimension: shape (l, B, T, d)
        values_stack = torch.stack(cached_values, dim=0)
        keys_stack = torch.stack(cached_keys, dim=0)
        
        # Compute dot product between q_l (d) and keys_stack (l, B, T, d) -> (l, B, T)
        logits = torch.einsum('d, l b t d -> l b t', self.q_l, keys_stack) * self.scale
        
        # Depth Softmax across layer history dimension (dim=0)
        alpha = F.softmax(logits, dim=0)  # (l, B, T)
        
        # Weighted combination of values: (l, B, T, 1) * (l, B, T, d) -> sum over l -> (B, T, d)
        x_l = torch.einsum('l b t, l b t d -> b t d', alpha, values_stack)
        
        return x_l, alpha
