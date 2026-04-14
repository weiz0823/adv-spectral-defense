import sys
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss

from adv_person.utils.bbox_utils import bbox_ious

_NO_CLS_SCORE_WARNED = True  # This warning is kept silent


def detection_loss(
    boxes: list[Tensor],
    targets: list[Tensor],
    iou_thresh: float,
    obj_cls_loss: Callable[[Tensor, Tensor], Tensor] = lambda obj, cls: obj,
    loss_type="max_conf",
    pad_zero=False,
    reduction="sum",
):
    """Compute detection loss over a batch of sample.

    Returns:
        loss: Tensor
    """
    global _NO_CLS_SCORE_WARNED
    all_losses = []
    for box, target in zip(boxes, targets):
        new_zero = lambda: torch.zeros(
            (), device=box.device, requires_grad=box.requires_grad
        )
        if len(box) > 0 and len(target) > 0:
            iou = bbox_ious(box, target)
            iou_max: Tensor = iou.max(dim=1)[0]  # Select closest target
            # Sift boxes that have a proper target
            keep = iou_max > iou_thresh
            valid_boxes = box[keep]
            if valid_boxes.shape[1] > 5:
                det_confs = obj_cls_loss(valid_boxes[:, 4], valid_boxes[:, 5])
            else:
                if not _NO_CLS_SCORE_WARNED:
                    print(
                        "[Warning] No classification score, ignoring obj_cls_loss.",
                        file=sys.stderr,
                    )
                    _NO_CLS_SCORE_WARNED = True
                det_confs = valid_boxes[:, 4]
            if len(det_confs) > 0:
                if loss_type == "max_conf":
                    # Default from Thys et al, 2019
                    all_losses.append(det_confs.max())
                elif loss_type == "max_iou":
                    # Used by Hu et al, 2023
                    valid_iou_max = iou_max[keep]
                    all_losses.append(det_confs[valid_iou_max.argmax()])
                elif loss_type == "sum_conf":
                    # Sum and mean both uncommon
                    all_losses.append(det_confs.sum())
                else:
                    raise ValueError(loss_type)
            elif pad_zero:
                all_losses.append(new_zero())
        elif pad_zero:
            all_losses.append(new_zero())
    if len(all_losses) > 0:
        loss_vec = torch.stack(all_losses)
        if reduction == "sum":
            return torch.sum(loss_vec)
        elif reduction == "mean":
            return torch.mean(loss_vec)
        elif reduction == "mean_test":
            return torch.sum(loss_vec) / len(boxes)
        elif reduction == "none":
            return loss_vec
        else:
            raise ValueError(reduction)
    else:
        return torch.zeros(())


class DetectionLoss(_Loss):
    def __init__(
        self,
        iou_thresh: float,
        obj_cls_loss: Callable[[Tensor, Tensor], Tensor] = lambda obj, cls: obj,
        reduction="mean",
        **kwargs,
    ) -> None:
        super().__init__(reduction=reduction)
        self.iou_thresh = iou_thresh
        self.obj_cls_loss = obj_cls_loss
        self.kwargs = kwargs

    def forward(self, boxes: list[Tensor], targets: list[Tensor]):
        return detection_loss(
            boxes,
            targets,
            self.iou_thresh,
            self.obj_cls_loss,
            reduction=self.reduction,
            **self.kwargs,
        )
