#!/usr/bin/env python
import argparse
import json
import os

from path import Path
import numpy as np
import torch
import yaml
from dataset.ds_contact_with_template import ContactWithTemplateDataset
from models.model_ESM2_with_prior import ESM2ContactWithTemplatePrior
from tqdm import tqdm

from models.utils import load_ckpt
from train import crop_item
from utils import upper_tri_mask, precision_at_k


def main(
        cfg: dict,
        split: str,
        ckpt: str,
):
    """Evaluate the model on the specified dataset split.
    Args:
        cfg: Configuration dictionary.
        split: Dataset split to evaluate on ("test").
        ckpt: Path to the model checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ContactWithTemplateDataset(
        cfg["data"]["processed_dir"],
        split,
        cfg["data"]["mmseqs_tsv"],
        top_k=cfg["model"]["top_k_templates"],
        bin_edges=cfg["model"]["distance_bins"]
    )

    model = ESM2ContactWithTemplatePrior(
        esm_model=cfg["model"]["esm_model"],
        freeze_esm=True,
        cnn_channels=cfg["model"]["cnn_channels"],
        cnn_depth=cfg["model"]["cnn_depth"],
        use_esm_contact_head=cfg["model"]["use_esm_contact_head"],
        num_dist_bins=len(cfg["model"]["distance_bins"]) + 1,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    load_ckpt(ckpt, model)

    model.eval()

    out_dir = Path(cfg["train"]["out_dir"]).expanduser() / f"pred_{split}"
    out_dir.makedirs_p()

    metrics = {"P@L": [], "P@L/2": [], "P@L/5": []}
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Eval {split}"):
            item = dataset[idx]
            seq = item["seq"]
            contact = torch.tensor(item["contact"], dtype=torch.float32, device=device)
            pri_contact = torch.tensor(item["pri_contact"], dtype=torch.float32, device=device)
            pri_bins = torch.tensor(item["pri_bins"], dtype=torch.float32, device=device)

            # crop if too long (avoiding OOM of the GPU)
            max_len = cfg["model"].get("max_len_eval")
            seq, contact, pri_contact, pri_bins = crop_item(seq, contact, pri_contact, pri_bins, max_len)
            seq_len = len(seq)

            # forward pass with autocast
            with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type=device.type):
                logits = model(seq, pri_contact, pri_bins)
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

    print("[EVAL]", summary)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="cfg/config.yaml", help="Path to config file")
    ap.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    ap.add_argument("--ckpt", type=str, default="model.ckpt", help="Path to model checkpoint")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.split, args.ckpt)
