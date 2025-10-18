import torch
from torch import Tensor


def precision_at_k(
        y_true: Tensor,
        y_score: Tensor,
        top_len: int,
) -> float:
    """
    Compute precision at k.
    Args:
        y_true: 1D tensor of true labels (0/1)
        y_score: 1D tensor of predicted scores
        top_len: int, number of top elements to consider
    Returns:
        precision at top_len
    """
    idx = torch.topk(y_score, k=min(top_len, y_score.numel()), largest=True).indices
    sel_true = y_true[idx]
    precision_at_k = sel_true.float().mean().item() if sel_true.numel() > 0 else 0.0
    return precision_at_k
