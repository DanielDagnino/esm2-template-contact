#!/usr/bin/env python
import argparse
import math
import os
import random

from path import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.ds_contact_with_template import ContactWithTemplateDataset
from models.model_ESM2_with_prior import ESM2ContactWithTemplatePrior
from utils import upper_tri_mask, precision_at_k
from scheduler.lr_lambda import get_scheduler
from optimizer.get_optimizer import get_optimizer
from cli.utils import collate_fn, crop_item


def main(
        cfg: dict
):
    """Main training loop.
    Args:
        cfg (dict): Configuration dictionary.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_set = ContactWithTemplateDataset(
        cfg["data"]["processed_dir"], 
        cfg["data"]["train_split"], 
        cfg["data"]["mmseqs_tsv"],
        top_k=cfg["model"]["top_k_templates"], 
        bin_edges=cfg["model"]["distance_bins"]
    )
    
    val_set = ContactWithTemplateDataset(
        cfg["data"]["processed_dir"], 
        cfg["data"]["val_split"], 
        cfg["data"]["mmseqs_tsv"],
        top_k=cfg["model"]["top_k_templates"], 
        bin_edges=cfg["model"]["distance_bins"]
    )

    # Loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=1, 
        shuffle=True, 
        num_workers=cfg["loader"].get("num_workers"), 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=cfg["loader"].get("num_workers"), 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2, 
        collate_fn=collate_fn
    )

    model = ESM2ContactWithTemplatePrior(
        esm_model=cfg["model"]["esm_model"], freeze_esm=cfg["model"]["freeze_esm"],
        cnn_channels=cfg["model"]["cnn_channels"], cnn_depth=cfg["model"]["cnn_depth"],
        use_esm_contact_head=cfg["model"]["use_esm_contact_head"],
        num_dist_bins=len(cfg["model"]["distance_bins"]) + 1, 
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # Optimizer and Scheduler
    opt = get_optimizer(model, cfg)
    sched = get_scheduler(opt, cfg)

    # Loss
    pos_weight = cfg["loss"].get("pos_weight")
    pos_weight = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    out_dir = Path(cfg["train"]["out_dir"]).expanduser()
    out_dir.makedirs_p()

    global_step = 0
    for epoch in range(999999):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            seq = batch["seq"]
            contact = batch["contact"].to(device, non_blocking=True)
            pri_contact = batch["pri_contact"].to(device, non_blocking=True)
            pri_bins = batch["pri_bins"].to(device, non_blocking=True)

            # crop if too long (avoiding OOM of the GPU)
            max_len = cfg["model"].get("max_len")
            seq, contact, pri_contact, pri_bins = crop_item(seq, contact, pri_contact, pri_bins, max_len)
            seq_len = len(seq)

            # training step
            model.train()
            global_step += 1

            # forward pass with autocast
            with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type=device.type):
                logits = model(seq, pri_contact, pri_bins)
                mask = upper_tri_mask(seq_len, sep_min=cfg["model"]["sep_min"], device=device)
                y = contact[mask]
                ylogits = logits[mask]
                loss = criterion(ylogits, y)

            # backward pass
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            # metrics
            with torch.no_grad():
                score = ylogits.sigmoid()

                k_len = int(seq_len)
                k_len2 = max(1, seq_len // 2)
                k_len5 = max(1, seq_len // 5)

                p_at_len = precision_at_k(y, score, k_len)
                p_at_len2 = precision_at_k(y, score, k_len2)
                p_at_len5 = precision_at_k(y, score, k_len5)

            pbar.set_postfix(loss=float(loss.item()), p_at_len=p_at_len, p_at_len2=p_at_len2, p_at_len5=p_at_len5)

            if global_step % cfg["train"]["eval_every"] == 0:
                evaluate(model, val_loader, device, cfg, sample_limit=64)

            if global_step % cfg["train"]["save_every"] == 0:
                ckpt = out_dir / f"step_{global_step}.pt"
                torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)

            if global_step >= cfg["optim"]["max_steps"]:
                print("Training complete.")
                evaluate(model, val_loader, device, cfg, sample_limit=64)
                return

        print(f"End of epoch {epoch}. Running evaluation ...")
        evaluate(model, val_loader, device, cfg, sample_limit=200)


def evaluate(model, loader, device, cfg, sample_limit):
    model.eval()
    metrics = {"P@L": [], "P@L/2": [], "P@L/5": []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (sample_limit is not None) and (i >= sample_limit):
                break
            seq = batch["seq"]
            contact = batch["contact"].to(device, non_blocking=True)
            pri_contact = batch["pri_contact"].to(device, non_blocking=True)
            pri_bins = batch["pri_bins"].to(device, non_blocking=True)

            # crop if too long (avoiding OOM of the GPU)
            max_len = cfg["model"].get("max_len")
            seq, contact, pri_contact, pri_bins = crop_item(seq, contact, pri_contact, pri_bins, max_len)
            seq_len = len(seq)

            logits = model(seq, pri_contact, pri_bins)
            mask = upper_tri_mask(seq_len, sep_min=cfg["model"]["sep_min"], device=device)
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

    print("[VAL]", summary)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    main(_cfg)
