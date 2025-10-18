#!/usr/bin/env python
import argparse
import json
from path import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset.ds_contact_with_template import ContactWithTemplateDataset
from models.model_ESM2_with_prior import ESM2ContactWithTemplatePrior
from models.utils import load_ckpt
from cli.utils import crop_item, upper_tri_mask


def main(
        cfg: dict,
        split: str,
        ckpt: str
):
    """Run prediction on a dataset split and dump NPZ files per-protein.
    Args:
        cfg: Configuration dictionary.
        split: Dataset split to predict on ('train', 'val', 'test').
        ckpt: Path to model checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ContactWithTemplateDataset(
        cfg["data"]["processed_dir"],
        split,
        cfg["data"]["mmseqs_tsv"],
        top_k=cfg["model"]["top_k_templates"],
        bin_edges=cfg["model"]["distance_bins"],
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

    out_dir = Path(cfg["train"]["out_dir"]).expanduser() / f"pred_{split}" / "npz"
    out_dir.makedirs_p()
    index_path = out_dir / "index.jsonl"

    with open(index_path, "w") as index_f, torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f"Predict {split}"):
            item = ds[i]
            name = item.get("name", f"protein_{i}")
            seq = item["seq"]
            contact = torch.tensor(item["contact"]).float().to(device)
            pri_contact = torch.tensor(item["pri_contact"]).float().to(device)
            pri_bins = torch.tensor(item["pri_bins"]).float().to(device)

            max_len = cfg["model"].get("max_len_eval")
            seq, contact, pri_contact, pri_bins = crop_item(seq, contact, pri_contact, pri_bins, max_len)
            L = len(seq)

            with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type=device.type):
                logits = model(seq, pri_contact, pri_bins)
                prob = logits.sigmoid()

            # Aux stats to support analysis stratifications
            mask_ulr = upper_tri_mask(L, sep_min=cfg["model"]["sep_min_eval"], device=device)
            density = float(contact[mask_ulr].mean().item()) if mask_ulr.sum() > 0 else 0.0
            t_strength = float(pri_contact.mean().item())  # simple proxy

            # Save NPZ (keep arrays on CPU)
            npz_path = out_dir / f"{name.replace('/', '_')}.npz"
            np.savez_compressed(
                npz_path,
                name=name,
                L=L,
                prob=prob.detach().cpu().numpy().astype("float32"),
                gt=contact.detach().cpu().numpy().astype("uint8"),
                density=np.array(density, dtype="float32"),
                template_strength=np.array(t_strength, dtype="float32"),
            )

            index_f.write(json.dumps({
                "name": name,
                "L": L,
                "npz": str(npz_path),
                "density": density,
                "template_strength": t_strength,
            }) + "\n")


if __name__ == "__main__":
    """
    Prediction dumper.

    - Loads the trained model and dataset split
    - Saves per-protein artifacts to NPZ for later analysis (to avoid recompute)

    Outputs:
      <out_dir>/pred_<split>/npz/<name>.npz  with keys:
        - name: str
        - L: int
        - prob: float32[L, L]  (sigmoid(logits))
        - gt:   uint8[L, L]    (binary GT contacts)
        - density: float32    (GT contact prevalence in upper-tri sep>=sep_min_eval)
        - template_strength: float32  (mean of pri_contact after crop)

    Also writes an index file:
      <out_dir>/pred_<split>/npz/index.jsonl
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    with open(args.config) as f:
       _cfg = yaml.safe_load(f)

    main(_cfg, args.split, args.ckpt)
