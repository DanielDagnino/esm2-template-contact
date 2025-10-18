#!/usr/bin/env python
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from path import Path
from tqdm import tqdm

from cli.utils_analysis import (
    separation_bins,
    make_sep_mask,
    precision_in_mask,
    prevalence_in_mask,
    pr_curve,
    plot_combined_precision_prevalence,
    plot_contacts_and_predictions,
    Loaded,
)


def main(
        pred_dir: str,
        out_dir: str,
        bin_size: int,
        max_sep: int,
        prob_thr: float,
        n_plot_contact_mat: int,
        sep_min: int,
):
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    (out_dir).makedirs_p()

    # Load all NPZs
    npz_files = sorted(list(pred_dir.files("*.npz")))
    loaded: List[Loaded] = []
    for f in tqdm(npz_files, desc="Load NPZ"):
        npz = np.load(f, allow_pickle=True)
        loaded.append(Loaded(
            name=str(npz["name"]).split("'")[-1] if isinstance(npz["name"], np.ndarray) else str(npz["name"]),
            L=int(npz["L"]),
            prob=np.array(npz["prob"], dtype=np.float32),
            gt=np.array(npz["gt"], dtype=np.uint8),
            density=float(npz["density"]),
            template_strength=float(npz["template_strength"]),
        ))

    # Separation binned precision & prevalence
    bins = separation_bins(max_sep=max_sep, bin_size=bin_size)
    bin_labels = [f"{lo}-{hi}" for (lo, hi) in bins]
    bin_precisions = [[] for _ in bins]
    bin_prevalences = [[] for _ in bins]

    # Range splits
    ranges = {
        "short": (6, 11),
        "medium": (12, 23),
        "long": (24, 10_000),
    }
    range_metrics = {k: {"P@L": [], "P@L/2": [], "P@L/5": [], "AUPRC": []} for k in ranges}

    # Global PR pools
    all_y = []
    all_s = []

    for item in tqdm(loaded, desc="Analyze"):
        L = item.L
        prob = item.prob
        gt = item.gt

        # Global PR accumulators (upper triangle sep>=1)
        I, J = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
        sep = np.abs(J - I)
        mask_ulr = (J > I) & (sep >= 1)
        all_y.append(gt[mask_ulr].astype(np.int32))
        all_s.append(prob[mask_ulr].astype(np.float64))

        # Separation bins
        for b_idx, (lo, hi) in enumerate(bins):
            m = make_sep_mask(L, lo, hi)
            if m.sum() == 0:
                continue
            bin_precisions[b_idx].append(precision_in_mask(gt, prob, m, thr=prob_thr))
            bin_prevalences[b_idx].append(prevalence_in_mask(gt, m))

        # Range metrics: compute top-k within range using k proportional to L
        for key, (lo, hi) in ranges.items():
            m = make_sep_mask(L, lo, hi)
            if m.sum() == 0:
                continue
            y_r = gt[m].astype(np.int32)
            s_r = prob[m].astype(np.float64)
            order = np.argsort(-s_r)
            k1 = int(L)
            k2 = max(1, L // 2)
            k5 = max(1, L // 5)
            p_at_l = float(y_r[order][:k1].mean()) if k1 <= len(order) else float('nan')
            p_at_l2 = float(y_r[order][:k2].mean()) if k2 <= len(order) else float('nan')
            p_at_l5 = float(y_r[order][:k5].mean()) if k5 <= len(order) else float('nan')
            range_metrics[key]["P@L"].append(p_at_l)
            range_metrics[key]["P@L/2"].append(p_at_l2)
            range_metrics[key]["P@L/5"].append(p_at_l5)
            p_curve, r_curve, auprc = pr_curve(y_r, s_r)
            range_metrics[key]["AUPRC"].append(float(auprc))

    # Aggregate
    agg_prec = [float(np.nanmean(v)) if len(v) else float('nan') for v in bin_precisions]
    agg_prev = [float(np.nanmean(v)) if len(v) else float('nan') for v in bin_prevalences]

    with open(out_dir / "precision_by_separation.json", "w") as f:
        json.dump({"bins": bin_labels, "precision": agg_prec}, f, indent=2)
    with open(out_dir / "prevalence_by_separation.json", "w") as f:
        json.dump({"bins": bin_labels, "prevalence": agg_prev}, f, indent=2)

    plot_combined_precision_prevalence(
        bin_labels, agg_prec, agg_prev,
        out_dir / "precision_prevalence_by_separation.png",
        title=f"Precision & prevalence vs separation (thr={prob_thr})",
    )

    # Global PR
    if len(all_y):
        y_cat = np.concatenate(all_y)
        s_cat = np.concatenate(all_s)
        p_curve, r_curve, auprc = pr_curve(y_cat, s_cat)
        np.savez(out_dir / "pr_curve_global.npz", precision=p_curve, recall=r_curve, auprc=auprc)
        plt.figure(figsize=(5,4))
        plt.plot(r_curve, p_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Global PR (AUPRC={auprc:.3f})')
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve_global.png", dpi=180)
        plt.close()

    # Range summaries
    range_summary = {}
    for key, d in range_metrics.items():
        range_summary[key] = {k: (float(np.nanmean(v)) if len(v) else float('nan')) for k, v in d.items()}
    with open(out_dir / "range_metrics.json", "w") as f:
        json.dump(range_summary, f, indent=2)

    # Best/Worst examples (by P@L) using evo-style plot
    scored = []
    for item in tqdm(loaded, desc="Plot best/worst examples"):
        L = item.L
        I, J = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
        sep = np.abs(J - I)
        mask = (J > I) & (sep >= 1)
        y = item.gt[mask].astype(np.int32)
        s = item.prob[mask].astype(np.float64)
        order = np.argsort(-s)
        k1 = int(L)
        p_at_l = float(y[order][:k1].mean()) if k1 <= len(order) else float('nan')
        scored.append((p_at_l, item))
    scored.sort(key=lambda t: (np.nan_to_num(t[0], nan=-1.0)))
    worst = scored[:n_plot_contact_mat]
    best = list(reversed(scored[-n_plot_contact_mat:]))

    for tag, group in [("best", best), ("worst", worst)]:
        gdir = out_dir / f"{tag}_examples"
        gdir.makedirs_p()
        for score, item in group:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()
            plot_contacts_and_predictions(item.prob, item.gt, sep_min, ax=ax, ms=1.5, title=True)
            out_path = gdir / f"{item.name.replace('/', '_')}_L{item.L}.png"
            plt.tight_layout()
            fig.savefig(out_path, dpi=180)
            plt.close(fig)

    # Compact summary
    summary = {
        "num_sequences": len(loaded),
        "bin_size": bin_size,
        "max_sep": max_sep,
        "prob_threshold": prob_thr,
    }
    with open(out_dir / "metrics_extended.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[ANALYZE] Done.")


if __name__ == '__main__':
    """
    Analysis-only.

    - Loads NPZ predictions dumped by predict.py
    - Produces:
      * Precision & GT Prevalence vs. separation
      * Long/Medium/Short-range metrics (P@L, P@L/2, P@L/5, AUPRC)
      * Global PR curve
      * Length stratification (<256, 256–512, >512)
      * Best/Worst 5 examples using evo-style
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--bin_size', type=int, default=32)
    ap.add_argument('--max_sep', type=int, default=640)
    ap.add_argument('--prob_thr', type=float, default=0.5)
    ap.add_argument('--n_plot_contact_mat', type=int, default=5)
    ap.add_argument('--sep_min', type=int, default=6)
    args = ap.parse_args()

    main(args.pred_dir, args.out_dir, args.bin_size, args.max_sep, args.prob_thr, args.n_plot_contact_mat, args.sep_min)
