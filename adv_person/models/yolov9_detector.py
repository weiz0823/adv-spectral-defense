import torch
from torch import Tensor
from torch.nn import Module

from adv_person.utils.bbox_utils import bbox_to_x1y1x2y2

from yolov9.models.common import DetectMultiBackend
from .i_detector import IDetector
from .nms import batched_nms


class YOLOv9Detector(DetectMultiBackend, IDetector):
    def get_bboxes(
        self,
        preds: Tensor,
        img_hw: tuple[int, int],
        conf=0.01,
        nms_thresh=0.7,
        n_mask=0,
    ):
        h, w = img_hw
        rv: list[tuple[Tensor, Tensor]] = []
        n_cls = preds.shape[1] - 4 - n_mask
        for pred in preds:
            boxes, cls, mask = pred.T.split([4, n_cls, n_mask], 1)
            scores, labels = torch.max(cls, dim=1)
            boxes: Tensor
            scores: Tensor
            labels: Tensor
            boxes = bbox_to_x1y1x2y2(boxes, "cxcywh")
            keep = scores > conf
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            keep = batched_nms(boxes, scores, labels, nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes / boxes.new_tensor((w, h, w, h))
            rv.append((torch.cat([boxes, scores.unsqueeze(1)], 1), labels))
        return rv

    def forward_test(self, images: Tensor, **kwargs):
        # preds: [N,K,6]. K predictions on each image. box,score,cls for each instance.
        # Box in [0,1]*[0,1].
        preds, train_out = self(images)
        return self.get_bboxes(preds, images.shape[-2:], **kwargs)
