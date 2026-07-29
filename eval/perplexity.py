import math
import torch
import torch.nn.functional as F
from models.gpt import GPT
from data.dataset import BinaryDatasetLoader

@torch.no_grad()
def evaluate_perplexity(model: GPT, dataloader: BinaryDatasetLoader, eval_iters: int = 100, device: str = 'cuda') -> tuple[float, float]:
    """
    Evaluates loss and cross-entropy perplexity over validation batches.
    """
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        X, Y = dataloader.get_batch()
        logits, loss = model(X, Y)
        losses.append(loss.item())
        
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, ppl
