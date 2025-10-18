from torch.nn import Module
from torch.optim import AdamW


def get_optimizer(
        model: Module,
        cfg: dict
) -> object:
    """
    Get optimizer from configuration.

    Args:
        model: The model to optimize.
        cfg: Configuration dictionary containing optimizer parameters.
    Returns:
        An optimizer instance.
    """

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"])
    )

    return optimizer
