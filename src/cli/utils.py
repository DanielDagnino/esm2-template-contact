#!/usr/bin/env python
import random
from typing import List

import torch
from torch import Tensor

from metrics.utils import precision_at_k
from gen_utils.general import upper_tri_mask


def collate_fn(
        items: List[dict],
):
    # items: list of dicts (we use batch_size=1)
    it = items[0]

    # Keep seq as str; convert numpy arrays to torch on CPU (not GPU yet)
    out = {
        "name": it["name"],
        "seq": it["seq"],
        "seq_len": it["seq_len"],
        "contact": torch.from_numpy(it["contact"]).float(),             # (seq_len, seq_len)
        "pri_contact": torch.from_numpy(it["pri_contact"]).float(),     # (seq_len, seq_len)
        "pri_bins": torch.from_numpy(it["pri_bins"]).float(),           # (seq_len, seq_len, B)
    }
    return out


# For GPU memory management, we can crop sequences longer than max_len
def crop_item(
        seq: str,
        contact: Tensor,
        pri_contact: Tensor,
        pri_bins: Tensor,
        max_len: int,
):
    """Crop the input item if its length exceeds max_len.
    Args:
        seq: str, amino acid sequence
        contact: Tensor of shape (seq_len, seq_len), contact map
        pri_contact: Tensor of shape (seq_len, seq_len), prior contact map
        pri_bins: Tensor of shape (seq_len, seq_len, B), prior distance bins
        max_len: int, maximum allowed length
    Returns:
        seq_c: str, cropped amino acid sequence
        contact_c: Tensor of shape (seq_len_c, seq_len_c), cropped contact map
        pri_contact_c: Tensor of shape (seq_len_c, seq_len_c), cropped prior contact map
        pri_bins_c: Tensor of shape (seq_len_c, seq_len_c, B), cropped prior distance bins
    """
    # Get the length of the sequence
    seq_len = len(seq)

    # If the sequence length is within the limit, return as is
    if seq_len <= max_len:
        return seq, contact, pri_contact, pri_bins

    # Randomly select a start position for cropping
    s = random.randint(0, seq_len - max_len)
    e = s + max_len

    # Crop the sequence and associated tensors
    seq_c = seq[s:e]
    contact_c = contact[s:e, s:e]
    pri_contact_c = pri_contact[s:e, s:e]
    pri_bins_c = pri_bins[s:e, s:e, :]

    return seq_c, contact_c, pri_contact_c, pri_bins_c
