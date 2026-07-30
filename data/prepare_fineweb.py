import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare_fineweb(output_dir: str = "data/fineweb1b", subset: str = "sample-10BT", target_tokens: int = 1_000_000_000):
    """
    Downloads FineWeb-Edu dataset (or sample), tokenizes with GPT-2 BPE tiktoken tokenizer,
    and streams tokenized uint16 directly to train.bin and val.bin.
    RAM usage stays below 500MB throughout tokenization.
    Split: 98% Train, 2% Validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        print(f"Data files already exist at {output_dir}. Skipping pre-tokenization.")
        return
        
    print(f"Downloading and pre-tokenizing FineWeb-Edu ({subset})...")
    try:
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)
    except Exception as e:
        print(f"Streaming failed: {e}. Falling back to standard dataset loading.")
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train")

    enc = tiktoken.get_encoding("gpt2")
    temp_bin_path = os.path.join(output_dir, "temp_all.bin")
    
    token_count = 0
    chunk = []
    chunk_size = 10_000_000  # Flush to disk every 10M tokens (~360MB RAM max)
    
    print(f"Encoding dataset into tokens (target: {target_tokens:,} tokens)...")
    with open(temp_bin_path, "wb") as f:
        for sample in tqdm(fw, desc="Tokenizing"):
            ids = enc.encode_ordinary(sample['text'])
            ids.append(enc.eot_token)
            chunk.extend(ids)
            token_count += len(ids)
            
            if len(chunk) >= chunk_size:
                arr = np.array(chunk, dtype=np.uint16)
                f.write(arr.tobytes())
                chunk.clear()
                
            if token_count >= target_tokens:
                break
                
        if len(chunk) > 0:
            arr = np.array(chunk, dtype=np.uint16)
            f.write(arr.tobytes())
            chunk.clear()

    # Split temp_all.bin into train.bin (98%) and val.bin (2%) using memmap
    all_data = np.memmap(temp_bin_path, dtype=np.uint16, mode='r')
    n = len(all_data)
    val_size = int(n * 0.02)
    train_size = n - val_size
    
    print(f"Total tokens tokenized: {n:,} | Train tokens: {train_size:,} | Val tokens: {val_size:,}")
    
    with open(train_bin_path, "wb") as f_train:
        for offset in range(0, train_size, chunk_size):
            end = min(offset + chunk_size, train_size)
            f_train.write(all_data[offset:end].tobytes())
            
    with open(val_bin_path, "wb") as f_val:
        for offset in range(train_size, n, chunk_size):
            end = min(offset + chunk_size, n)
            f_val.write(all_data[offset:end].tobytes())

    all_data._mmap.close()
    del all_data
    if os.path.exists(temp_bin_path):
        os.remove(temp_bin_path)

    print(f"Successfully saved {train_bin_path} and {val_bin_path} with low RAM footprint!")


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
