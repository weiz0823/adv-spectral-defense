import torch
from torch import Tensor

# from torch.utils.data._utils.collate import default_collate


def detection_collate(batch: list[tuple[Tensor, Tensor]]):
    images = []
    targets = []
    target_lens = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
        target_lens.append(len(tgt))
    return (
        torch.stack(images),
        torch.cat(targets),
        torch.tensor(target_lens, dtype=torch.int32),
    )
