import os
import tempfile
import numpy as np
import pytest
from data.prepare_fineweb import generate_synthetic_data
from data.dataset import BinaryDatasetLoader

def test_synthetic_data_generation_and_loading():
    """Verify synthetic dataset generation and memory-mapped reader."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_synthetic_data(tmp_dir, num_tokens=1000)
        
        train_path = os.path.join(tmp_dir, "train.bin")
        val_path = os.path.join(tmp_dir, "val.bin")
        
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        
        loader = BinaryDatasetLoader(tmp_dir, 'train', block_size=16, batch_size=4, device='cpu')
        x, y = loader.get_batch()
        
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        assert (y[:, :-1] == x[:, 1:]).all(), "Targets y must be x shifted by 1 token position"
        loader.close()

