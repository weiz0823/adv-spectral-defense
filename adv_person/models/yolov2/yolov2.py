import os
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import Module

from adv_person.utils.bbox_utils import bbox_to_x1y1x2y2

from ..i_detector import IDetector
from .darknet import Darknet


class YOLOV2(Module, IDetector):
    anchors: Tensor

    def __init__(
        self,
        conf_thresh=0.01,
        cfgfile: Optional[str] = None,
        weightfile: Optional[str] = None,
        obj_cls_loss: Callable[[Tensor, Tensor], Tensor] = lambda obj, cls: obj * cls,
        backbone_cls=Darknet,
    ) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh
        if cfgfile is None:
            cfgfile = os.path.join(os.path.dirname(__file__), "yolov2.cfg")
        self.darknet_model = backbone_cls(cfgfile)
        if weightfile is not None:
            self.darknet_model.load_weights(weightfile)
        self.num_classes = self.darknet_model.num_classes
        self.num_anchors = self.darknet_model.num_anchors
        anchor_tensor = torch.tensor(self.darknet_model.anchors).reshape(
            self.num_anchors, -1
        )
        self.register_buffer("anchors", anchor_tensor, False)
        self.anchor_step = len(self.anchors) / self.num_anchors
        self.obj_cls_loss = obj_cls_loss

    def forward(self, x: Tensor):
        return self.extract_feat(x)

    def forward_train(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        self.darknet_model(x)
        return self.darknet_model.loss

    def extract_feat(self, x: Tensor) -> Tensor:
        return self.darknet_model(x)

    def forward_adv_yolo(self, x: Tensor):
        output: Tensor = self.darknet_model(x)
        batch, channels, h, w = output.shape
        assert channels == (5 + self.num_classes) * self.num_anchors

        output = output.view(batch * self.num_anchors, 5 + self.num_classes, h * w)
        output = output.transpose(0, 1)
        output = output.reshape(5 + self.num_classes, batch, self.num_anchors, h * w)

        det_confs = torch.sigmoid(output[4])
        cls_confs = output[5:].softmax(dim=0)
        cls_conf_person = cls_confs[0]
        loss = self.obj_cls_loss(det_confs, cls_conf_person)
        return torch.max(loss, dim=1)[0]

    def forward_test(
        self, x: Tensor, conf_thresh: Optional[float] = None
    ) -> list[tuple[Tensor, Tensor]]:
        return self.get_bboxes(self.darknet_model(x), conf_thresh=conf_thresh)

    def get_bboxes(
        self, output: Tensor, conf_thresh: Optional[float] = None, box_size=7
    ):
        """Get bounding boxes with class indices.

        Args:
            box_size:
                4: loc
                5: (loc,det_conf)
                6: (loc,det_conf,class_conf)
                7: (loc,det_conf,class_conf),class_index
        """
        assert 4 <= box_size and box_size <= 7
        device = output.device
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch, channels, h, w = output.shape
        assert channels == (5 + self.num_classes) * self.num_anchors

        output = output.view(batch * self.num_anchors, 5 + self.num_classes, h * w)
        output = output.transpose(0, 1)
        output = output.reshape(5 + self.num_classes, batch, self.num_anchors, h * w)
        grid_x, grid_y = torch.meshgrid(
            [torch.arange(w, device=device), torch.arange(h, device=device)],
            indexing="xy",
        )
        xs = torch.sigmoid(output[0]) + grid_x.flatten()
        ys = torch.sigmoid(output[1]) + grid_y.flatten()

        anchor_w = self.anchors[:, 0:1]
        anchor_h = self.anchors[:, 1:2]
        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h

        # batch, num_anchors, h*w
        det_confs = torch.sigmoid(output[4])
        conf = det_confs.flatten(1)
        inds = conf > (conf_thresh or self.conf_thresh)

        boxes_cxcy = torch.stack((xs / w, ys / h, ws / w, hs / h), dim=-1)
        boxes_xyxy = bbox_to_x1y1x2y2(boxes_cxcy, "cxcywh")

        if box_size >= 6:
            cls_confs = output[5:].softmax(dim=0)
            cls_max_confs: Tensor
            cls_max_ids: Tensor
            cls_max_confs, cls_max_ids = torch.max(cls_confs, dim=0)
            # FIXME: this is to pass person class confidence
            cls_max_confs = cls_confs[0]

        # batch, num_anchors, h*w, 6
        if box_size == 4:
            boxes_with_score = boxes_xyxy
        elif box_size == 5:
            boxes_with_score = torch.cat((boxes_xyxy, det_confs.unsqueeze(-1)), dim=-1)
        else:  # box_size >= 6
            boxes_with_score = torch.cat(
                (boxes_xyxy, det_confs.unsqueeze(-1), cls_max_confs.unsqueeze(-1)),
                dim=-1,
            )
        boxes_with_score = boxes_with_score.flatten(1, 2)
        if box_size == 7:
            cls_max_ids = cls_max_ids.flatten(1)

        # For each sample, return a tuple of boxes with score and class indices
        if box_size <= 6:
            det_results = [box[i] for box, i in zip(boxes_with_score, inds)]
        else:
            det_results = [
                (box[i], ind[i])
                for box, ind, i in zip(boxes_with_score, cls_max_ids, inds)
            ]
        return det_results
