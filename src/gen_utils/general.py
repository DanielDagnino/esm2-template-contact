import torch
from torch import Tensor


def upper_tri_mask(
        seq_len: int,
        sep_min: int = 6,
        device: torch.device = None,
) -> Tensor:
    """Generate upper-triangular mask with minimum sequence separation.
    Args:
        seq_len, sequence length
        sep_min, minimum sequence separation
        device, device for the output mask
    Returns:
        mask: tensor of shape (L, L)
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (j > i) & ((j - i) >= sep_min)
    return mask
