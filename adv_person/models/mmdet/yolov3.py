from typing import Optional

from mmdet.models import DETECTORS, YOLOV3
from torch import Tensor

from ..i_detector import IDetector


@DETECTORS.register_module()
class YOLOV3Detector(YOLOV3, IDetector):
    """Decorator to set some patch forward functions."""

    def extract_feat(self, img: Tensor) -> tuple[Tensor, ...]:
        return super().extract_feat(img)

    def get_bboxes(
        self, feat: tuple[Tensor, ...], img_metas: list[dict], rescale=False
    ) -> list[tuple[Tensor, Tensor]]:
        return self.bbox_head.simple_test(feat, img_metas, rescale=rescale)

    def forward_test(
        self, img: Tensor, img_metas: Optional[list[dict]] = None, rescale=True
    ):
        if img_metas is None:
            scale = img.shape[2]
            scale = (scale, scale, scale, scale)
            img_metas = [dict(scale_factor=scale) for _ in range(len(img))]
        return self.get_bboxes(self.extract_feat(img), img_metas, rescale)
