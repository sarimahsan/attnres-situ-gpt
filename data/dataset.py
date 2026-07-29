import os
import numpy as np
import torch

class BinaryDatasetLoader:
    """
    High-performance memory-mapped binary dataset reader (nanoGPT style).
    Reads uint16 pre-tokenized bin files.
    """
    def __init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device: str):
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        bin_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Binary data file {bin_path} not found. Run python data/prepare_fineweb.py first!"
            )
            
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)

    def close(self):
        if hasattr(self, 'data') and self.data is not None:
            if hasattr(self.data, '_mmap') and self.data._mmap is not None:
                self.data._mmap.close()
            del self.data
            self.data = None

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if 'cuda' in self.device:
            # Pin memory & copy asynchronously to GPU
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            
        return x, y

