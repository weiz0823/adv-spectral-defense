from typing import Optional

from mmcv.runner import force_fp32
from mmdet.core.utils import select_single_mlvl
from mmdet.models import DETECTORS, FCOS, HEADS, FCOSHead, RetinaNet
from torch import Tensor

from ..i_detector import IDetector


@HEADS.register_module()
class FCOSHeadExtend(FCOSHead):
    @force_fp32(apply_to=("cls_scores", "bbox_preds"))
    def get_bboxes(
        self,
        cls_scores,
        bbox_preds,
        score_factors=None,
        img_metas=None,
        cfg=None,
        rescale=False,
        with_nms=True,
        **kwargs,
    ):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, False)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, False)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id, False)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                score_factor_list,
                mlvl_priors,
                img_meta,
                cfg,
                rescale,
                with_nms,
                **kwargs,
            )
            result_list.append(results)
        return result_list


@DETECTORS.register_module()
class FCOSDetector(FCOS, IDetector):
    def extract_feat(self, img: Tensor):
        return super().extract_feat(img)

    def get_bboxes(
        self, feat, img_metas: list[dict], rescale=False
    ) -> list[tuple[Tensor, Tensor]]:
        return self.bbox_head.simple_test(feat, img_metas, rescale=rescale)

    def forward_test(
        self,
        img: Tensor,
        img_metas: Optional[list[dict]] = None,
        rescale=True,
    ):
        if img_metas is None:
            h, w = img.shape[2:]
            img_metas = [
                dict(scale_factor=(w, h, w, h), img_shape=(h, w))
                for _ in range(len(img))
            ]
        return self.get_bboxes(self.extract_feat(img), img_metas, rescale)


@DETECTORS.register_module()
class RetinaNetDetector(RetinaNet, IDetector):
    def extract_feat(self, img: Tensor):
        return super().extract_feat(img)

    def get_bboxes(
        self, feat, img_metas: list[dict], rescale=False
    ) -> list[tuple[Tensor, Tensor]]:
        return self.bbox_head.simple_test(feat, img_metas, rescale=rescale)

    def forward_test(
        self,
        img: Tensor,
        img_metas: Optional[list[dict]] = None,
        rescale=True,
    ):
        if img_metas is None:
            h, w = img.shape[2:]
            img_metas = [
                dict(scale_factor=(w, h, w, h), img_shape=(h, w))
                for _ in range(len(img))
            ]
        return self.get_bboxes(self.extract_feat(img), img_metas, rescale)
