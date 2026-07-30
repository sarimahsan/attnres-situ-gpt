import os
import sys
import json
import argparse

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.model_config import GPTConfig
from config.train_config import TrainConfig
from data.prepare_fineweb import prepare_fineweb, generate_synthetic_data
from training.trainer import Trainer

VARIANTS = ["baseline", "attn_res", "situ_glu", "both"]
SEEDS = [42, 1337, 2024]
MANIFEST_PATH = "output/experiments_manifest.json"

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    
    # Initialize fresh manifest for 12 runs
    manifest = {}
    for v in VARIANTS:
        for s in SEEDS:
            key = f"{v}_seed{s}"
            manifest[key] = {
                "variant": v,
                "seed": s,
                "status": "PENDING", # PENDING, IN_PROGRESS, COMPLETED, FAILED
                "summary": None
            }
    save_manifest(manifest)
    return manifest

def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

def print_status(manifest: dict):
    print("\n" + "=" * 65)
    print("      EXPERIMENT MATRIX STATUS (4 Variants x 3 Seeds)")
    print("=" * 65)
    completed = 0
    total = len(manifest)
    for key, info in manifest.items():
        status = info["status"]
        if status == "COMPLETED":
            completed += 1
            val_loss = info.get("summary", {}).get("final_val_loss", "N/A")
            val_ppl = info.get("summary", {}).get("final_val_ppl", "N/A")
            print(f"  [✓] {key:<20} | Status: COMPLETED | Val Loss: {val_loss} | Val PPL: {val_ppl}")
        elif status == "IN_PROGRESS":
            print(f"  [➔] {key:<20} | Status: IN_PROGRESS")
        else:
            print(f"  [ ] {key:<20} | Status: {status}")
    print("-" * 65)
    print(f"Progress: {completed}/{total} runs completed ({(completed/total)*100:.1f}%)\n")

def run_single_experiment(variant: str, seed: int, data_dir: str, target_tokens: int, max_steps: int = None, synthetic: bool = False):
    manifest = load_manifest()
    run_key = f"{variant}_seed{seed}"
    
    if run_key not in manifest:
        raise ValueError(f"Unknown run key {run_key}")
        
    if manifest[run_key]["status"] == "COMPLETED":
        print(f"Run {run_key} is already COMPLETED. Skipping.")
        return

    print(f"\nLaunching Run: {run_key} (Variant={variant}, Seed={seed})...")
    manifest[run_key]["status"] = "IN_PROGRESS"
    save_manifest(manifest)

    # Ensure dataset is ready
    if synthetic:
        generate_synthetic_data(data_dir)
    else:
        prepare_fineweb(data_dir, target_tokens=target_tokens)

    model_config = GPTConfig(variant=variant)
    train_config = TrainConfig(
        current_seed=seed,
        target_tokens=target_tokens if max_steps is None else 0,
        max_iters=max_steps if max_steps is not None else 5000
    )

    try:
        trainer = Trainer(model_config, train_config, data_dir, run_key)
        summary = trainer.train()
        
        manifest[run_key]["status"] = "COMPLETED"
        manifest[run_key]["summary"] = summary
        save_manifest(manifest)
    except Exception as e:
        manifest[run_key]["status"] = "FAILED"
        save_manifest(manifest)
        print(f"Run {run_key} FAILED with error: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for 12-Run Experiment Matrix")
    parser.add_argument("--variant", type=str, choices=VARIANTS, help="Specific variant to run")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Specific seed to run")
    parser.add_argument("--run-all-pending", action="store_true", help="Run all PENDING experiments in manifest sequentially")
    parser.add_argument("--data-dir", type=str, default="data/fineweb1b")
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000, help="Target token budget per run (default 1B)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps (for testing/dry-runs)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for rapid local testing")
    parser.add_argument("--status", action="store_true", help="Display current experiment matrix status and exit")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.status:
        print_status(manifest)
        return

    if args.run_all_pending:
        print("Running all PENDING experiments in manifest...")
        for key, info in manifest.items():
            if info["status"] in ("PENDING", "FAILED"):
                run_single_experiment(
                    info["variant"], info["seed"], 
                    args.data_dir, args.target_tokens, 
                    args.max_steps, args.synthetic
                )
        print_status(load_manifest())
    elif args.variant and args.seed:
        run_single_experiment(
            args.variant, args.seed, 
            args.data_dir, args.target_tokens, 
            args.max_steps, args.synthetic
        )
        print_status(load_manifest())
    else:
        print_status(manifest)
        print("Specify --variant and --seed to launch a run, or --run-all-pending to execute all.")

if __name__ == "__main__":
    main()
