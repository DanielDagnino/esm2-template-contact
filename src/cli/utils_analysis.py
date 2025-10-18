#!/usr/bin/env python
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from path import Path
from tqdm import tqdm


@dataclass
class Loaded:
    name: str
    L: int
    prob: np.ndarray  # (L,L)
    gt: np.ndarray    # (L,L)
    density: float
    template_strength: float


def separation_bins(max_sep: int = 512, bin_size: int = 32) -> List[Tuple[int, int]]:
    bins = []
    start = 0
    while start <= max_sep:
        end = start + bin_size - 1
        bins.append((start, end))
        start += bin_size
    return bins


def make_sep_mask(L: int, sep_lo: int, sep_hi: int) -> np.ndarray:
    I, J = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    sep = np.abs(J - I)
    m = (J > I) & (sep >= sep_lo) & (sep <= sep_hi)
    return m


def precision_in_mask(gt: np.ndarray, prob: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    sel = mask
    if sel.sum() == 0:
        return float('nan')
    y = gt[sel].astype(np.float32)
    p = (prob[sel] >= thr).astype(np.float32)
    tp = float((p * y).sum())
    fp = float((p * (1.0 - y)).sum())
    if tp + fp == 0:
        return float('nan')
    return tp / (tp + fp)


def prevalence_in_mask(gt: np.ndarray, mask: np.ndarray) -> float:
    sel = mask
    if sel.sum() == 0:
        return float('nan')
    y = gt[sel].astype(np.float32)
    return float(y.mean())


def pr_curve(gt: np.ndarray, prob: np.ndarray):
    y = gt.reshape(-1).astype(np.int32)
    s = prob.reshape(-1).astype(np.float64)
    order = np.argsort(-s)
    y_sorted = y[order]

    P = float(y.sum())
    if P == 0:
        return np.array([1.0]), np.array([0.0]), float('nan')

    tp = 0.0
    fp = 0.0
    precisions = []
    recalls = []
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / P)
    p = np.asarray(precisions)
    r = np.asarray(recalls)
    order2 = np.argsort(r)
    r2 = r[order2]
    p2 = p[order2]
    auprc = float(np.trapz(p2, r2))
    return p2, r2, auprc


def compute_long_range_pl(pred: np.ndarray, gt: np.ndarray, minsep: int = 24) -> float:
    L = gt.shape[0]
    I, J = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    sep = np.abs(J - I)
    mask = (J > I) & (sep >= minsep)
    y = gt[mask].astype(np.int32)
    s = pred[mask].astype(np.float64)
    if y.size == 0:
        return float('nan')
    order = np.argsort(-s)
    k = L
    return float(y[order][:k].mean())


def plot_combined_precision_prevalence(bin_labels, precision, prevalence, out_path, title):
    """Combined precision & prevalence bar plot."""
    fig, ax1 = plt.subplots(figsize=(11, 4))
    x = np.arange(len(bin_labels))
    ax1.bar(x - 0.2, precision, width=0.4, label='Precision')
    ax1.set_ylabel('Precision')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, prevalence, width=0.4, label='Prevalence', alpha=0.6)
    ax2.set_ylabel('GT prevalence')
    ax2.set_ylim(0, max(0.05, min(0.5, (max(prevalence) if len(prevalence) else 0.1) * 1.2)))
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.set_title(title)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_contacts_and_predictions(
    predictions: np.ndarray,
    contacts: np.ndarray,
    sep_min: int,
    ax=None,
    cmap: str = "Blues",
    ms: float = 1.0,
    title=True,
    animated: bool = False,
):
    """Single-panel contact viz adapted from:
        https://github.com/rmrao/evo/blob/main/evo/visualize.py
        https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
    - Upper triangle:
        - blue = TP predicted contact in top-L
        - red = FP predicted contact in top-L
        - grey = Contact not predicted in top-L
    - Lower triangle: predicted probabilities
    """
    if ax is None:
        ax = plt.gca()

    pred = np.array(predictions, dtype=np.float64)
    gt = np.array(contacts, dtype=np.uint8)
    L = gt.shape[0]

    # Build masks
    rel = np.add.outer(-np.arange(L), np.arange(L))
    bottom_mask = rel < 0
    masked_image = np.ma.masked_where(bottom_mask, pred)

    # Invalidate < sep_min separation for selection of top-L
    invalid_mask = (np.abs(np.add.outer(np.arange(L), -np.arange(L))) < sep_min)
    pred_for_topL = pred.copy()
    pred_for_topL[invalid_mask] = float('-inf')

    # Top-L predictions
    topl_val = np.sort(pred_for_topL.reshape(-1))[-L]
    pred_contacts = pred_for_topL >= topl_val

    true_positives = (gt.astype(bool) & pred_contacts & ~bottom_mask)
    false_positives = (~gt.astype(bool) & pred_contacts & ~bottom_mask)
    false_negatives = (gt.astype(bool) & ~pred_contacts & ~bottom_mask)

    # Title handling
    if isinstance(title, str):
        title_text = title
    elif title:
        long_pl = compute_long_range_pl(pred, gt, minsep=24)
        if callable(title):
            title_text = title(long_pl)
        else:
            title_text = f"Long Range P@L: {100.0*long_pl:0.1f}" if not np.isnan(long_pl) else "Long Range P@L: n/a"
    else:
        title_text = None

    img = ax.imshow(masked_image, cmap=cmap, animated=animated)
    ax.plot(*np.where(false_negatives), "o", c="grey", ms=ms)[0]
    ax.plot(*np.where(false_positives), "o", c="r", ms=ms)[0]
    ax.plot(*np.where(true_positives), "o", c="b", ms=ms)[0]
    if title_text is not None:
        ax.set_title(title_text)

    ax.axis("square")
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    return img
