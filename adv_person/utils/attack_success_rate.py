from typing import Optional

import numpy as np
import torch
from torch import Tensor

from adv_person.utils.bbox_utils import bbox_ious


class AttackSuccessRate(object):
    def __init__(
        self,
        iou_threshold: float = 0.5,
        conf_thresholds: Optional[list[float]] = None,
        use_cpu=True,
    ) -> None:
        self.iou_threshold = iou_threshold
        if conf_thresholds is None:
            self.conf_thresholds = np.arange(0.1, 0.99, 0.1).tolist()
        else:
            self.conf_thresholds = conf_thresholds
        self.use_cpu = use_cpu
        self.all_conf_scores: list[Tensor] = []  # maximum confidence of each target

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]):
        for pred, tgt in zip(preds, target):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            tgt_boxes = tgt["boxes"]
            if self.use_cpu:
                pred_boxes = pred_boxes.cpu()
                pred_scores = pred_scores.cpu()
                tgt_boxes = tgt_boxes.cpu()
            all_scores = tgt_boxes.new_zeros((len(tgt_boxes),))
            iou = bbox_ious(tgt_boxes, pred_boxes)
            iou_selects = iou > self.iou_threshold
            for i, iou_sel in enumerate(iou_selects.unbind(0)):
                scores = pred_scores[iou_sel]
                if len(scores) > 0:
                    all_scores[i] = scores.max().item()
            self.all_conf_scores.append(all_scores)

    def compute(self) -> dict[str, Tensor]:
        rv = {}
        success_rates = []
        all_conf_scores = torch.cat(self.all_conf_scores)
        for conf in self.conf_thresholds:
            success_rates.append(
                torch.sum(all_conf_scores <= conf) / len(all_conf_scores)
            )
        # print(success_rates)
        rv["masr"] = sum(success_rates) / len(success_rates)
        hardcode_confs = (0, 0.01, 0.1, 0.25, 0.5, 0.75)
        for conf in hardcode_confs:
            rv[f"asr{int(conf * 100.0)}"] = torch.sum(all_conf_scores <= conf) / len(
                all_conf_scores
            )
        return rv
