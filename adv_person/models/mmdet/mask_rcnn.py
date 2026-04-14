from typing import Optional

from mmdet.models import DETECTORS, MaskRCNN
from torch import Tensor

from ..i_detector import IDetector


@DETECTORS.register_module()
class MaskRCNNDetector(MaskRCNN, IDetector):
    def extract_feat(self, img: Tensor):
        return super().extract_feat(img)

    def get_bboxes(
        self, feat, img_metas: list[dict], proposals=None, rescale=False
    ) -> list[tuple[Tensor, Tensor]]:
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feat, img_metas)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale
        )
        return list(zip(det_bboxes, det_labels))

    def forward_test(
        self,
        img: Tensor,
        img_metas: Optional[list[dict]] = None,
        proposals=None,
        rescale=True,
    ):
        if img_metas is None:
            h, w = img.shape[2:]
            img_metas = [
                dict(scale_factor=(w, h, w, h), img_shape=(h, w))
                for _ in range(len(img))
            ]
        return self.get_bboxes(self.extract_feat(img), img_metas, proposals, rescale)
