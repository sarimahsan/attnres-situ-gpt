from dataclasses import dataclass
from typing import Literal

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    
    # Variant configuration
    variant: Literal['baseline', 'attn_res', 'situ_glu', 'both'] = 'baseline'
    
    # SiTU-GLU Hyperparameters (Kimi K3 Technical Report Eq. 12)
    situ_beta1: float = 4.0
    situ_beta2: float = 25.0
    
    # Multiplier for GLU hidden dim (LLaMA style 8/3 scaling)
    mlp_multiple_of: int = 64
    
    @property
    def is_attn_res(self) -> bool:
        return self.variant in ('attn_res', 'both')
        
    @property
    def is_situ_glu(self) -> bool:
        return self.variant in ('situ_glu', 'both')
