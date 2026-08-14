import os
import time
import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn

from config.model_config import GPTConfig
from config.train_config import TrainConfig
from models.gpt import GPT
from data.dataset import BinaryDatasetLoader
from training.logger import ExperimentLogger

class LossSpikeDetector:
    """Tracks rolling mean and std to count loss spikes."""
    def __init__(self, window_size: int = 50, std_threshold: float = 3.0):
        self.history = deque(maxlen=window_size)
        self.std_threshold = std_threshold
        self.spike_count = 0

    def update(self, loss_val: float) -> bool:
        is_spike = False
        if len(self.history) >= 20:
            mean = float(np.mean(self.history))
            std = float(np.std(self.history))
            if std > 1e-6 and loss_val > (mean + self.std_threshold * std):
                self.spike_count += 1
                is_spike = True
        self.history.append(loss_val)
        return is_spike


class Trainer:
    def __init__(self, model_config: GPTConfig, train_config: TrainConfig, data_dir: str, run_name: str):
        self.m_cfg = model_config
        self.t_cfg = train_config
        self.data_dir = data_dir
        self.run_dir = os.path.join(train_config.out_dir, run_name)
        
        # Prevent CUDA memory fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')

        # Initialize model
        raw_model = GPT(model_config)
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and 'cuda' in train_config.device:
            print(f"Detected {torch.cuda.device_count()} GPUs! Enabling DataParallel across GPUs.", flush=True)
            self.model = torch.nn.DataParallel(raw_model).to(train_config.device)
        else:
            self.model = raw_model.to(train_config.device)

        self.raw_model = raw_model
        
        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=train_config.weight_decay,
            learning_rate=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            device_type='cuda' if 'cuda' in train_config.device else 'cpu'
        )
        
        self.train_loader = BinaryDatasetLoader(
            data_dir, 'train', train_config.block_size, train_config.batch_size, train_config.device
        )
        self.val_loader = BinaryDatasetLoader(
            data_dir, 'val', train_config.block_size, train_config.batch_size, train_config.device
        )
        
        self.logger = ExperimentLogger(self.run_dir)
        self.spike_detector = LossSpikeDetector()
        
        # Mixed Precision AMP setup (BF16, no loss scaling needed)
        device_type = 'cuda' if 'cuda' in train_config.device else 'cpu'
        self.ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        
        # Metrics and counters
        self.flops_per_token = self.raw_model.estimate_flops_per_token()
        self.tokens_per_step = train_config.batch_size * train_config.block_size * train_config.gradient_accumulation_steps
        self.cumulative_tokens = 0
        self.cumulative_flops = 0.0
        self.nan_inf_count = 0
        
        # Compute max iterations based on target token budget
        if train_config.target_tokens > 0:
            self.max_iters = train_config.target_tokens // self.tokens_per_step
        else:
            self.max_iters = train_config.max_iters

    def get_lr(self, it: int) -> float:
        """Cosine learning rate decay with linear warmup."""
        if not self.t_cfg.decay_lr:
            return self.t_cfg.learning_rate
        if it < self.t_cfg.warmup_iters:
            return self.t_cfg.learning_rate * it / self.t_cfg.warmup_iters
        if it > self.max_iters:
            return self.t_cfg.min_lr
        decay_ratio = (it - self.t_cfg.warmup_iters) / (self.max_iters - self.t_cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.t_cfg.min_lr + coeff * (self.t_cfg.learning_rate - self.t_cfg.min_lr)

    @torch.no_grad()
    def estimate_val_loss(self) -> tuple[float, float]:
        """Runs validation loss and computes perplexity."""
        self.model.eval()
        losses = torch.zeros(self.t_cfg.eval_iters)
        for k in range(self.t_cfg.eval_iters):
            X, Y = self.val_loader.get_batch()
            with self.ctx:
                _, loss, _ = self.model(X, Y, return_logits=False)
                if loss.ndim > 0:
                    loss = loss.mean()
            losses[k] = loss.item()
        val_loss = losses.mean().item()
        val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
        self.model.train()
        return val_loss, val_ppl

    def train(self) -> dict:
        """Main training loop."""
        self.model.train()
        start_time = time.time()
        best_val_loss = float('inf')
        
        # Log config
        config_summary = {
            "model_variant": self.m_cfg.variant,
            "seed": self.t_cfg.current_seed,
            "num_params": self.raw_model.get_num_params(),
            "flops_per_token": self.flops_per_token,
            "target_tokens": self.t_cfg.target_tokens,
            "max_iters": self.max_iters,
            "situ_beta1": self.m_cfg.situ_beta1,
            "situ_beta2": self.m_cfg.situ_beta2,
        }
        self.logger.log_config(config_summary)

        print(f"Starting training run: {os.path.basename(self.run_dir)}", flush=True)
        print(f"Params: {self.raw_model.get_num_params():,} | FLOPs/tok: {self.flops_per_token:e} | Total Iters: {self.max_iters}", flush=True)
        print(f"Precision: bfloat16 | GradScaler: Disabled", flush=True)

        for iter_num in range(1, self.max_iters + 1):
            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Gradient Accumulation
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            step_act_max = 0.0
            
            for micro_step in range(self.t_cfg.gradient_accumulation_steps):
                X, Y = self.train_loader.get_batch()
                with self.ctx:
                    _, loss, act_max_batch = self.model(X, Y, return_logits=False)
                    if loss.ndim > 0:
                        loss = loss.mean()
                    loss = loss / self.t_cfg.gradient_accumulation_steps
                loss_accum += loss.item()
                if act_max_batch is not None:
                    step_act_max = max(step_act_max, float(act_max_batch.max().item()))
                loss.backward()

            # Parameter Norms & Diagnostics
            param_norms = [p.detach().norm(2).item() for p in self.model.parameters()]
            param_norm_mean = float(np.mean(param_norms)) if param_norms else 0.0
            param_norm_max = float(np.max(param_norms)) if param_norms else 0.0
            act_max = step_act_max

            # Gradient Clipping (Computes total global norm and scales gradients before optimizer step)
            if self.t_cfg.grad_clip != 0.0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t_cfg.grad_clip)
                grad_norm_val = float(total_grad_norm.item() if hasattr(total_grad_norm, 'item') else total_grad_norm)
            else:
                grads = [p.grad.detach().norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None]
                grad_norm_val = math.sqrt(sum(grads)) if grads else 0.0

            # Check for non-finite values (NaN / Inf)
            is_non_finite = not math.isfinite(loss_accum) or not math.isfinite(grad_norm_val)
            if is_non_finite:
                self.nan_inf_count += 1

            # Optimizer Step (Always after gradient clipping)
            self.optimizer.step()

            # Update step metrics
            self.cumulative_tokens += self.tokens_per_step
            self.cumulative_flops += self.flops_per_token * self.tokens_per_step
            self.spike_detector.update(loss_accum)
            wall_clock = time.time() - start_time
            tok_per_sec = self.cumulative_tokens / wall_clock if wall_clock > 0 else 0.0

            # Approximate time remaining (ETA)
            rem_iters = max(0, self.max_iters - iter_num)
            eta_sec = rem_iters * (wall_clock / iter_num) if iter_num > 0 else 0.0
            if eta_sec < 60:
                eta_str = f"{int(eta_sec)}s"
            elif eta_sec < 3600:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
            else:
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60):02d}m"

            # Live Step Progress output every 10 steps (or step 1) with rich diagnostics
            if iter_num % self.t_cfg.log_interval == 0 or iter_num == 1:
                pct = (iter_num / self.max_iters) * 100
                print(f"Step {iter_num}/{self.max_iters} ({pct:.1f}%) | Loss: {loss_accum:.4f} | LR: {lr:.2e} | GradNorm(pre-clip): {grad_norm_val:.2f} | ActMax: {act_max:.2f} | Speed: {tok_per_sec:,.0f} tok/s | ETA: {eta_str}", flush=True)

            # Evaluation Interval
            val_loss, val_ppl = "", ""
            if iter_num % self.t_cfg.eval_interval == 0 or iter_num == self.max_iters:
                v_loss, v_ppl = self.estimate_val_loss()
                val_loss, val_ppl = v_loss, v_ppl
                print(f"==> Step {iter_num}/{self.max_iters} EVAL | Train Loss: {loss_accum:.4f} | Val Loss: {v_loss:.4f} | Val PPL: {v_ppl:.2f}", flush=True)

                # Save best model checkpoint
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    if self.t_cfg.always_save_checkpoint:
                        torch.save(self.raw_model.state_dict(), os.path.join(self.run_dir, "best.pt"))

            # Save latest checkpoint
            if (iter_num % self.t_cfg.eval_interval == 0 or iter_num == self.max_iters) and self.t_cfg.always_save_checkpoint:
                torch.save(self.raw_model.state_dict(), os.path.join(self.run_dir, "latest.pt"))

            # Log metrics to CSV
            if iter_num % self.t_cfg.log_interval == 0 or iter_num == self.max_iters:
                self.logger.log_step({
                    "step": iter_num,
                    "train_loss": round(loss_accum, 5),
                    "val_loss": round(val_loss, 5) if val_loss != "" else "",
                    "val_ppl": round(val_ppl, 2) if val_ppl != "" else "",
                    "grad_norm_mean": round(grad_norm_val, 5),
                    "grad_norm_max": round(grad_norm_val, 5),
                    "param_norm_mean": round(param_norm_mean, 5),
                    "param_norm_max": round(param_norm_max, 5),
                    "act_max": round(act_max, 4),
                    "scaler_scale": 1.0,
                    "nan_inf_count": self.nan_inf_count,
                    "loss_spikes": self.spike_detector.spike_count,
                    "lr": round(lr, 8),
                    "tokens_per_sec": round(tok_per_sec, 1),
                    "wall_clock_sec": round(wall_clock, 2),
                    "cumulative_flops": f"{self.cumulative_flops:.6e}",
                    "cumulative_tokens": self.cumulative_tokens,
                })

        final_val_loss, final_val_ppl = self.estimate_val_loss()
        summary = {
            "variant": self.m_cfg.variant,
            "seed": self.t_cfg.current_seed,
            "final_train_loss": round(loss_accum, 5),
            "final_val_loss": round(final_val_loss, 5),
            "final_val_ppl": round(final_val_ppl, 2),
            "best_val_loss": round(best_val_loss, 5),
            "total_spikes": self.spike_detector.spike_count,
            "total_wall_clock_sec": round(time.time() - start_time, 2),
            "total_tokens_seen": self.cumulative_tokens,
            "total_flops": f"{self.cumulative_flops:.6e}",
        }
        self.logger.log_summary(summary)
        print(f"Run completed successfully: {self.m_cfg.variant} seed={self.t_cfg.current_seed} | Final Val Loss: {final_val_loss:.4f} PPL: {final_val_ppl:.2f}")
        return summary
