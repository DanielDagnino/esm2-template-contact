import torch


def load_ckpt(ckpt_path, model):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
