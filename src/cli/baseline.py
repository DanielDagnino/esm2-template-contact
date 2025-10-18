#!/usr/bin/env python
import argparse
import json
import os

from path import Path
import esm
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dataset.ds_contact_with_template import ContactWithTemplateDataset
from train import crop_item
from utils import upper_tri_mask, precision_at_k


class ESMContactBaseline(nn.Module):
    """
    Thin wrapper around the pretrained ESM-1b contact predictor to serve as a
    baseline. It ignores template priors and any custom CNN heads.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        # ESM-1b has a built-in contact prediction head shipped with fair-esm
        self.esm_model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()
        self.esm_model.to(device)
        self._device = device

    @torch.no_grad()
    def forward(self, seq: str, *_, **__):
        """
        Returns contact *logits* of shape [seq_len, seq_len] for a single sequence.
        Extra args are accepted (and ignored) for drop-in compatibility with the
        previous custom model signature.
        """
        _, _, tokens = self.batch_converter([("protein", seq)])  # [1, T]
        tokens = tokens.to(self._device)
        # return_contacts=True yields a (B, seq_len, seq_len) prob matrix without special tokens
        out = self.esm_model(tokens, repr_layers=[], return_contacts=True)
        probs = out["contacts"][0]  # [seq_len, seq_len], values in [0,1]
        # Convert probabilities to logits so downstream code (sigmoid) still works
        probs = probs.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(probs) - torch.log1p(-probs)
        return logits


def main(cfg, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ContactWithTemplateDataset(
        cfg["data"]["processed_dir"],
        split,
        cfg["data"]["mmseqs_tsv"],
        top_k=cfg["model"]["top_k_templates"],
        bin_edges=cfg["model"]["distance_bins"],
    )

    # Baseline model: ESM contact predictor
    model = ESMContactBaseline(device=device)
    model.eval()

    out_dir = Path(cfg["train"]["out_dir"]).expanduser() / f"pred_baseline_esm"
    out_dir.makedirs_p()

    metrics = {"P@L": [], "P@L/2": [], "P@L/5": []}
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Eval {split} (ESM baseline)"):
            item = dataset[i]
            seq = item["seq"]
            contact = torch.tensor(item["contact"], dtype=torch.float32, device=device)
            pri_contact = torch.tensor(item["pri_contact"], dtype=torch.float32, device=device)
            pri_bins = torch.tensor(item["pri_bins"], dtype=torch.float32, device=device)

            # crop if too long (avoiding OOM of the GPU)
            max_len = cfg["model"].get("max_len_eval")
            seq, contact, pri_contact, pri_bins = crop_item(seq, contact, pri_contact, pri_bins, max_len)
            seq_len = len(seq)

            # forward pass with autocast
            with torch.amp.autocast(dtype=torch.float16, enabled=device.type == "cuda", device_type=device.type):
                logits = model(seq, pri_contact, pri_bins)  # [seq_len, seq_len]
                mask = upper_tri_mask(seq_len, sep_min=cfg["model"]["sep_min_eval"], device=device)
                y = contact[mask]
                ylogits = logits[mask].sigmoid()

            metrics["P@L"].append(precision_at_k(y, ylogits, int(seq_len)))
            metrics["P@L/2"].append(precision_at_k(y, ylogits, max(1, seq_len // 2)))
            metrics["P@L/5"].append(precision_at_k(y, ylogits, max(1, seq_len // 5)))

    summary = {
        k: float(np.mean(v))
        if len(v) > 0
        else 0.0
        for k, v in metrics.items()
    }

    print("[EVAL-ESM-Baseline]", summary)
    with open(out_dir / "baseline.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="cfg/config.yaml", help="Path to config file")
    ap.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.split)
