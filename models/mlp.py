import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import GPTConfig

class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Calculate hidden dim (LLaMA style 8/3 * n_embd)
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        if config.mlp_multiple_of:
            hidden_dim = config.mlp_multiple_of * ((hidden_dim + config.mlp_multiple_of - 1) // config.mlp_multiple_of)
            
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x W_g) * (x W_u)) W_d
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return self.dropout(out)


class SiTUGLU(nn.Module):
    """
    Sigmoid Tanh Unit GLU (SiTU-GLU) from Kimi K3 Technical Report (Eq. 12).
    
    Formula:
    SiTU-GLU(x) = [ beta_1 * tanh( (x W_g)/beta_1 ) * Sigmoid(x W_g) ] 
                  * [ beta_2 * tanh( (x W_u)/beta_2 ) ] W_d
                  
    Guarantees bounded pre-down-projection activations: |Activation| <= beta_1 * beta_2 (default 100).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        if config.mlp_multiple_of:
            hidden_dim = config.mlp_multiple_of * ((hidden_dim + config.mlp_multiple_of - 1) // config.mlp_multiple_of)
            
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        self.beta1 = config.situ_beta1  # Default 4.0
        self.beta2 = config.situ_beta2  # Default 25.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_g = self.w_gate(x)
        z_u = self.w_up(x)
        
        # Gate branch: beta_1 * tanh(z_g / beta_1) * Sigmoid(z_g)
        gate = self.beta1 * torch.tanh(z_g / self.beta1) * torch.sigmoid(z_g)
        
        # Up branch: beta_2 * tanh(z_u / beta_2)
        up = self.beta2 * torch.tanh(z_u / self.beta2)
        
        out = self.w_down(gate * up)
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.is_situ_glu:
            self.net = SiTUGLU(config)
        else:
            self.net = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
