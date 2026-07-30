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
        ix = np.random.randint(0, self.n_tokens - self.block_size - 1, size=self.batch_size)
        offsets = ix[:, None] + np.arange(self.block_size)
        
        x_np = self.data[offsets].astype(np.int64)
        y_np = self.data[offsets + 1].astype(np.int64)
        
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        
        if 'cuda' in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            
        return x, y

