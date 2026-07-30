import os
import csv
import json
from typing import Dict, Any

class ExperimentLogger:
    """
    Structured logger for tracking model metrics, FLOPs, activation maxima,
    gradient stability, parameter norms, and AMP scaler states.
    Saves metrics directly to CSV and summary JSON.
    """
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self.summary_path = os.path.join(run_dir, "summary.json")
        self.config_path = os.path.join(run_dir, "config.json")
        
        self.fieldnames = [
            "step", "train_loss", "val_loss", "val_ppl", 
            "grad_norm_mean", "grad_norm_max", 
            "param_norm_mean", "param_norm_max",
            "act_max", "scaler_scale", "nan_inf_count",
            "loss_spikes", "lr", "tokens_per_sec", "wall_clock_sec", 
            "cumulative_flops", "cumulative_tokens"
        ]
        
        # Initialize CSV file with headers
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_config(self, config_dict: Dict[str, Any]):
        """Save model and training configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def log_step(self, metrics: Dict[str, Any]):
        """Log a training step's metrics to CSV."""
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            row = {field: metrics.get(field, "") for field in self.fieldnames}
            writer.writerow(row)

    def log_summary(self, summary_dict: Dict[str, Any]):
        """Save final run summary."""
        with open(self.summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)
