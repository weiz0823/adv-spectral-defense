from .i_detector import *
import torch
from torch import nn, Tensor
from torch.nn import Module
from typing import Any
import rfdetr
from rfdetr.detr import RFDETR

from adv_person.utils.bbox_utils import bbox_to_x1y1x2y2


class RFDETRDetector(Module, IDetector):
    def __init__(self, type_="RFDETRSmall", **kwargs: Any) -> None:
        super().__init__()
        wrapper_cls = getattr(rfdetr, type_)
        wrapper: RFDETR = wrapper_cls(**kwargs)
        self.model = wrapper.model.model

    def forward_test(self, images: Tensor, **kwargs) -> list[tuple[Tensor, Tensor]]:
        outputs: dict[str, Tensor] = self.model(images)
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        prob, labels = out_logits.sigmoid().max(dim=-1)
        boxes_xyxy = bbox_to_x1y1x2y2(out_bbox, "cxcywh")
        all_boxes: list[tuple[Tensor, Tensor]] = []
        for box, score, label in zip(boxes_xyxy, prob, labels):
            all_boxes.append((torch.cat([box, score.unsqueeze(-1)], dim=1), label))
        return all_boxes
