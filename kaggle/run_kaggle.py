"""
Kaggle Execution Script for AttnRes + SiTU-GLU Research Framework
------------------------------------------------------------------
Run this script inside a Kaggle GPU notebook cell:
!python kaggle/run_kaggle.py --variant baseline --seed 42 --target-tokens 500000000
"""

import os
import sys
import subprocess
import argparse

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_cmd(cmd: str):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU Runner")
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "attn_res", "situ_glu", "both"])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 1337, 2024])
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for quick testing")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest unit tests before starting training")
    args = parser.parse_args()

    print("==========================================================")
    print("      Kaggle GPU Bootstrapper - SSF AttnRes & SiTU-GLU")
    print("==========================================================")

    # 1. Run unit test suite if requested
    if args.run_tests:
        print("\n--- Phase 1: Running Unit Tests ---")
        run_cmd("pytest tests/")

    # 2. Execute target experiment run
    print(f"\n--- Phase 2: Running Variant '{args.variant}' Seed {args.seed} ---")
    synth_flag = "--synthetic" if args.synthetic else ""
    run_cmd(f"python scripts/run_matrix.py --variant {args.variant} --seed {args.seed} --target-tokens {args.target-tokens} {synth_flag}")

    # 3. Render Analysis and Figures
    print("\n--- Phase 3: Generating Statistical Figures and Tables ---")
    run_cmd("python scripts/analyze_results.py")

    print("\n==========================================================")
    print("  Kaggle run completed successfully! Output figures saved.")
    print("==========================================================")

if __name__ == "__main__":
    main()
