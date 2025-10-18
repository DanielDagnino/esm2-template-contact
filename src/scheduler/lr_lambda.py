import math

from torch.optim.lr_scheduler import LambdaLR


def get_linear_warmup_cosine(
        warmup: int,
        total: int
):
    """ Linear warmup + cosine decay scheduler
    Args:
        warmup: number of warmup steps
        total: total number of steps
    Returns:
        A function that takes in the current step and returns the learning rate multiplier
    """
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        # cosine from 1 to 0
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    return fn


def get_scheduler(
        opt,
        cfg
):
    sched = LambdaLR(
        opt,
        lr_lambda=get_linear_warmup_cosine(cfg["optim"]["warmup_steps"], cfg["optim"]["max_steps"])
    )
    return sched
