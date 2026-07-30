import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VARIANTS = ["baseline", "attn_res", "situ_glu", "both"]
VARIANT_LABELS = {
    "baseline": "0. Baseline (Pre-Norm + SwiGLU)",
    "attn_res": "1. +AttnRes",
    "situ_glu": "2. +SiTU",
    "both":     "3. +Both (AttnRes + SiTU)"
}
COLORS = {
    "baseline": "#1f77b4", # Blue
    "attn_res": "#ff7f0e", # Orange
    "situ_glu": "#2ca02c", # Green
    "both":     "#d62728"  # Red
}

def parse_experiment_data(runs_dir: str = "output/runs"):
    """Collect metric CSVs and summary JSONs across all completed runs."""
    results = {v: [] for v in VARIANTS}
    loss_curves = {v: [] for v in VARIANTS}
    grad_norms = {v: [] for v in VARIANTS}
    
    for v in VARIANTS:
        pattern = os.path.join(runs_dir, f"{v}_seed*", "summary.json")
        summary_files = glob.glob(pattern)
        
        for sf in summary_files:
            run_dir = os.path.dirname(sf)
            csv_file = os.path.join(run_dir, "metrics.csv")
            
            with open(sf, "r") as f:
                summary = json.load(f)
                results[v].append(summary)
                
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                loss_curves[v].append(df[["step", "train_loss", "val_loss"]].dropna(subset=["train_loss"]))
                grad_norms[v].append(df[["step", "grad_norm_mean", "grad_norm_max"]].dropna(subset=["grad_norm_mean"]))

    return results, loss_curves, grad_norms

def generate_mock_data():
    """Generates realistic mock data for demonstrating analysis pipeline."""
    results = {}
    loss_curves = {}
    grad_norms = {}
    steps = np.arange(10, 5010, 10)
    
    base_loss_curve = 3.5 * np.exp(-steps / 1000.0) + 2.8
    
    for v_idx, v in enumerate(VARIANTS):
        results[v] = []
        loss_curves[v] = []
        grad_norms[v] = []
        
        # Offset performance slightly per variant for demonstration
        benefit = 0.0 if v == "baseline" else (0.08 if v == "attn_res" else (0.06 if v == "situ_glu" else 0.12))
        
        for seed in [42, 1337, 2024]:
            noise = np.random.normal(0, 0.01, size=len(steps))
            t_loss = base_loss_curve - benefit + noise
            v_loss = t_loss[-1] + np.random.normal(0.02, 0.005)
            
            results[v].append({
                "variant": v,
                "seed": seed,
                "final_val_loss": round(v_loss, 4),
                "final_val_ppl": round(float(np.exp(v_loss)), 2),
                "total_spikes": 2 if v in ("baseline", "attn_res") else 0,
                "num_params": 50_234_880 if "attn_res" not in v else 50_237_952,
                "flops_per_token": "3.01e+08",
            })
            
            df_loss = pd.DataFrame({"step": steps, "train_loss": t_loss, "val_loss": t_loss + 0.02})
            loss_curves[v].append(df_loss)
            
            # Gradient norms (SiTU variants exhibit lower max spike bounds)
            g_mean = 0.4 + np.random.normal(0, 0.05, size=len(steps))
            g_max = 1.2 + (np.random.exponential(1.5, size=len(steps)) if "situ" not in v else np.random.exponential(0.3, size=len(steps)))
            df_grad = pd.DataFrame({"step": steps, "grad_norm_mean": g_mean, "grad_norm_max": g_max})
            grad_norms[v].append(df_grad)
            
    return results, loss_curves, grad_norms

def plot_loss_curves(loss_curves: dict, out_path: str = "output/Figure1_Training_Loss_Curves.png"):
    """Generates Figure 1: Training Loss Curves Overlaid Across Variants."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)

    for v in VARIANTS:
        if not loss_curves[v]:
            continue
        
        # Calculate mean loss across seeds per step
        dfs = loss_curves[v]
        min_len = min(len(df) for df in dfs)
        steps = dfs[0]["step"].values[:min_len]
        losses = np.array([df["train_loss"].values[:min_len] for df in dfs])
        
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)
        
        plt.plot(steps, mean_loss, label=VARIANT_LABELS[v], color=COLORS[v], linewidth=2.0)
        plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, color=COLORS[v], alpha=0.15)
        
    plt.title("Figure 1: FineWeb-Edu Training Loss Curves (50M LM)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Cross Entropy Training Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Figure 1 to {out_path}")

def plot_gradient_stability(grad_norms: dict, out_path: str = "output/Figure2_Gradient_Stability.png"):
    """Generates Figure 2: Gradient Norm / Stability Comparison."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    for v in VARIANTS:
        if not grad_norms[v]:
            continue
            
        dfs = grad_norms[v]
        min_len = min(len(df) for df in dfs)
        steps = dfs[0]["step"].values[:min_len]
        
        means = np.array([df["grad_norm_mean"].values[:min_len] for df in dfs]).mean(axis=0)
        maxs = np.array([df["grad_norm_max"].values[:min_len] for df in dfs]).mean(axis=0)
        
        ax1.plot(steps, means, label=VARIANT_LABELS[v], color=COLORS[v], linewidth=1.5)
        ax2.plot(steps, maxs, label=VARIANT_LABELS[v], color=COLORS[v], linewidth=1.5, alpha=0.85)

    ax1.set_title("(a) Mean Gradient Norm per Step", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Mean Grad Norm")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)
    
    ax2.set_title("(b) Max Gradient Norm per Step (Stability Signal)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Max Grad Norm")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.suptitle("Figure 2: Gradient Norm Stability & Activation Bounding Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Figure 2 to {out_path}")

def render_table_1(results: dict) -> str:
    """Generates Markdown Table 1: Validation Loss/PPL, Mean +- Std over 3 seeds."""
    rows = []
    headers = "| # | Variant Name | Params | FLOPs/Tok | Final Val Loss (Mean ± Std) | Val Perplexity | Loss Spikes |"
    divider = "|---|---|---|---|---|---|---|"
    
    for idx, v in enumerate(VARIANTS):
        seed_data = results[v]
        if not seed_data:
            rows.append(f"| {idx} | {VARIANT_LABELS[v]} | - | - | N/A | N/A | N/A |")
            continue
            
        losses = [d["final_val_loss"] for d in seed_data]
        ppls = [d["final_val_ppl"] for d in seed_data]
        spikes = [d["total_spikes"] for d in seed_data]
        params = seed_data[0]["num_params"]
        flops = seed_data[0]["flops_per_token"]
        
        mean_l, std_l = np.mean(losses), np.std(losses)
        mean_p, std_p = np.mean(ppls), np.std(ppls)
        mean_spk = np.mean(spikes)
        
        rows.append(f"| {idx} | {VARIANT_LABELS[v]} | {params:,} | {flops} | {mean_l:.4f} ± {std_l:.4f} | {mean_p:.2f} ± {std_p:.2f} | {mean_spk:.1f} |")

    table_md = "\n".join([headers, divider] + rows)
    return table_md

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="output/runs")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--mock", action="store_true", help="Generate analysis from mock data if runs are incomplete")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mock or not os.path.exists(args.runs_dir):
        print("Using mock data to render initial analysis figures and tables...")
        results, loss_curves, grad_norms = generate_mock_data()
    else:
        results, loss_curves, grad_norms = parse_experiment_data(args.runs_dir)

    fig1_path = os.path.join(args.output_dir, "Figure1_Training_Loss_Curves.png")
    fig2_path = os.path.join(args.output_dir, "Figure2_Gradient_Stability.png")
    table1_path = os.path.join(args.output_dir, "results_table1.md")

    # Render Figures
    plot_loss_curves(loss_curves, fig1_path)
    plot_gradient_stability(grad_norms, fig2_path)

    # Render Table 1
    t1_md = render_table_1(results)
    print("\n" + "=" * 80)
    print("Table 1: Main Quantitative Results Across 4 Variants (3 Seeds Each)")
    print("=" * 80)
    print(t1_md)
    print("=" * 80 + "\n")
    
    with open(table1_path, "w") as f:
        f.write("# Main Quantitative Results (Table 1)\n\n" + t1_md + "\n")
        
    print(f"Analysis script completed successfully. Output saved in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
