import abc

from torch import Tensor


class IDetector:
    """Detector interface for patch training."""

    def extract_feat(self, images: Tensor):
        raise NotImplementedError

    def get_bboxes(self, output: Tensor, **kwargs):
        raise NotImplementedError

    def forward_test(self, images: Tensor, **kwargs) -> list[tuple[Tensor, Tensor]]:
        raise NotImplementedError
