import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare_fineweb(output_dir: str = "data/fineweb1b", num_proc: int = 4, subset: str = "sample-10BT"):
    """
    Downloads FineWeb-Edu dataset (or sample), tokenizes with GPT-2 BPE tiktoken tokenizer,
    and writes train.bin and val.bin as uint16 binary files.
    Split: 98% Train, 2% Validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        print(f"Data files already exist at {output_dir}. Skipping pre-tokenization.")
        return
        
    print(f"Downloading and pre-tokenizing FineWeb-Edu ({subset})...")
    # Load dataset from HuggingFace
    try:
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)
    except Exception as e:
        print(f"Streaming failed: {e}. Falling back to standard dataset loading.")
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train")

    enc = tiktoken.get_encoding("gpt2")

    def process_doc(example):
        tokens = enc.encode_ordinary(example['text'])
        tokens.append(enc.eot_token)
        return {'ids': tokens, 'len': len(tokens)}

    # Tokenize tokens and save to binary memory-mapped files
    print("Encoding dataset into tokens...")
    all_tokens = []
    max_tokens_target = 1_000_000_000  # ~1B tokens target subset
    
    token_count = 0
    for sample in tqdm(fw, desc="Tokenizing"):
        ids = enc.encode_ordinary(sample['text'])
        ids.append(enc.eot_token)
        all_tokens.extend(ids)
        token_count += len(ids)
        if token_count >= max_tokens_target:
            break

    arr = np.array(all_tokens, dtype=np.uint16)
    n = len(arr)
    
    # 98 / 2 train / val split
    val_size = int(n * 0.02)
    train_arr = arr[:-val_size]
    val_arr = arr[-val_size:]
    
    print(f"Total tokens: {n:,} | Train tokens: {len(train_arr):,} | Val tokens: {len(val_arr):,}")
    
    train_arr.tofile(train_bin_path)
    val_arr.tofile(val_bin_path)
    print(f"Successfully saved {train_bin_path} and {val_bin_path}!")


def generate_synthetic_data(output_dir: str = "data/fineweb1b", num_tokens: int = 10_000_000):
    """Fallback generator for local testing / dry-runs when offline."""
    os.makedirs(output_dir, exist_ok=True)
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        return

    print(f"Generating synthetic token dataset ({num_tokens:,} tokens) for testing...")
    arr = np.random.randint(0, 50257, size=num_tokens, dtype=np.uint16)
    val_size = int(num_tokens * 0.02)
    train_arr = arr[:-val_size]
    val_arr = arr[-val_size:]
    
    train_arr.tofile(train_bin_path)
    val_arr.tofile(val_bin_path)
    print(f"Synthetic data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/fineweb1b")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic tokens for testing")
    args = parser.parse_args()
    
    if args.synthetic:
        generate_synthetic_data(args.out_dir)
    else:
        prepare_fineweb(args.out_dir)
