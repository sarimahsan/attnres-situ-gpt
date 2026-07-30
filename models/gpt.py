import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import GPTConfig
from models.attention import CausalSelfAttention
from models.mlp import MLP
from models.attn_res import AttnResRouter, LayerRMSNorm

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. PyTorch standard. """
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.last_act_max = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln_1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        with torch.no_grad():
            self.last_act_max = max(float(attn_out.abs().max().item()), float(mlp_out.abs().max().item()))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Output language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (GPT-2 style)
        self.transformer.wte.weight = self.lm_head.weight

        # Attention Residual Routers & RMSNorm key-cachers if +AttnRes variant
        if config.is_attn_res:
            self.attn_res_routers = nn.ModuleList([
                AttnResRouter(i, config) for i in range(config.n_layer)
            ])
            self.history_rms_norm = LayerRMSNorm(config.n_embd)

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w_down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        By default non-embedding parameters are counted (wpe is counted, wte/lm_head tied are excluded).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def estimate_flops_per_token(self) -> float:
        """
        Estimate theoretical FLOPs per token (forward + backward pass).
        Standard Transformer: ~6 * N_params + 12 * n_layer * n_head * d_head * sequence_len.
        AttnRes: Adds depth-attention routing FLOPs.
        """
        N = self.get_num_params(non_embedding=False)
        L = self.config.n_layer
        H = self.config.n_head
        Q = self.config.n_embd // H
        T = self.config.block_size
        
        # Base matrix multiplication FLOPs per token (6N)
        base_flops = 6.0 * N
        # Causal Attention KV sequence FLOPs
        seq_attn_flops = 12.0 * L * H * Q * T
        
        # AttnRes depth routing FLOPs: sum_{l=1}^L (2 * l * d) = L*(L+1)*d FLOPs per token
        attn_res_flops = (L * (L + 1) * self.config.n_embd) if self.config.is_attn_res else 0.0
        
        return base_flops + seq_attn_flops + attn_res_flops

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # (t)

        # Forward token & position embeddings
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (t, n_embd)
        h0 = self.transformer.drop(tok_emb + pos_emb)

        if not self.config.is_attn_res:
            # Baseline / SiTU: standard sequential pre-norm residual stream
            x = h0
            for block in self.transformer.h:
                x = block(x)
        else:
            # +AttnRes / +Both: Full Attention Residual depth-routing with incremental key caching
            cached_values = [h0]
            cached_keys = [self.history_rms_norm(h0)]
            
            x_current = h0
            for i, (block, router) in enumerate(zip(self.transformer.h, self.attn_res_routers)):
                # Route input x_i from layer history [y_0, ..., y_{i-1}] using cached keys v_j = RMSNorm(y_j)
                x_routed, alpha_weights = router(cached_values, cached_keys)
                # Forward block transformation
                y_i = block(x_routed)
                
                # Cache output value y_i and cached key v_i = RMSNorm(y_i)
                cached_values.append(y_i)
                cached_keys.append(self.history_rms_norm(y_i))
                x_current = y_i

            x = x_current

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only project the last token's logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        with torch.no_grad():
            self.last_act_max = max(getattr(b, 'last_act_max', 0.0) for b in self.transformer.h)

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay = set()
        no_decay = set()
        seen_param_ids = set()

        for pn, p in param_dict.items():
            if id(p) in seen_param_ids:
                continue
            seen_param_ids.add(id(p))

            if p.dim() < 2 or pn.endswith('bias') or 'wte' in pn or 'wpe' in pn or 'ln_' in pn or 'norm' in pn:
                no_decay.add(pn)
            else:
                decay.add(pn)

        # Validate parameter partitioning
        inter_params = decay & no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
