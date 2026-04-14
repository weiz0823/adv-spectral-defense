import torch
from torch import Tensor
from torch.utils.data import Subset

# eps for fp32, which ensures 1+eps>1
_EPS_FP32 = 1e-5
# eps for fp16
_EPS_FP16 = 1e-3


def get_eps(x: Tensor):
    """Get eps according to the dtype of `x`.

    Args:
        x (`Tensor`):
            Used to get the dtype.

    Returns:
        `float`: The corresponding eps.
    """
    return torch.finfo(x.dtype).eps


def gen_uniform_grid(
    pmin: list[float], pmax: list[float], pnum: list[int], flatten=False, **kwargs
):
    """Input min, max, num of each axis, return grid points from A x B x C."""
    axes = [torch.linspace(m, M, n, **kwargs) for m, M, n in zip(pmin, pmax, pnum)]
    if flatten:
        return torch.cartesian_prod(*axes)
    else:
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def gen_uniform_grid_arange(pnum: list[int], flatten=False, **kwargs):
    """Input min, max, num of each axis, return grid points from A x B x C."""
    axes = [torch.arange(_, **kwargs) for _ in pnum]
    if flatten:
        return torch.cartesian_prod(*axes)
    else:
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def acc1(logits: Tensor, y: Tensor):
    """Top 1 accuracy.

    Args:
        logits ([N, K] Tensor): logits.
        y ([N] int Tensor): labels.

    Returns:
        accuracy ([] float Tensor): accuracy on the batch.
    """
    return (logits.argmax(dim=1) == y).float().mean()


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred: Tensor
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res: list[Tensor] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append(correct_k / batch_size)
    return res


def count_parameters(model: torch.nn.Module):
    """Count number of parameters in the model.

    Args:
        model (Module): ...

    Returns:
        int: number of parameters.
    """
    return sum(
        v.numel()
        for name, v in model.named_parameters()
        if "auxiliary_head" not in name
    )


def unwrap_dataset(dataset):
    """Unwrap dataset subset.

    Args:
        dataset (Dataset): dataset, probably a subset.

    Returns:
        dataset (Dataset): the subset unwrapped.
    """
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset
