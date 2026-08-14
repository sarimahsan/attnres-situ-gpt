import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    out_dir: str = 'output/runs'
    eval_interval: int = 2000
    log_interval: int = 10
    eval_iters: int = 100
    always_save_checkpoint: bool = False
    
    # Batch & Token parameters (Optimized for 2x T4 15GB GPUs)
    batch_size: int = 32           # micro-batch size per forward (~8GB VRAM per GPU on DataParallel)
    block_size: int = 512          # context length
    gradient_accumulation_steps: int = 4   # 32 * 512 * 4 = 65,536 tokens per step
    
    # Target Token Budget (e.g. 1B tokens)
    target_tokens: int = 1_000_000_000
    
    # Optimization
    learning_rate: float = 1.0e-4  # Robust learning rate preventing loss divergence
    max_iters: int = 5000          # will be computed dynamically based on target_tokens if set
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 1500
    lr_decay_iters: int = 5000
    min_lr: float = 1.0e-5
    
    # Experiment Matrix Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])
    current_seed: int = 42
    
    # System settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16'
    
    def get_tokens_per_step() -> int:
        return self.batch_size * self.block_size * self.gradient_accumulation_steps
