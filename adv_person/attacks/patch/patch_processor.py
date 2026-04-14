from typing import Optional

from torch import Tensor
from torch.nn import Module

from .patch_applier import PatchApplier
from .patch_transformer import PatchTransformer
from .random_crop import PatchCropping
from .thin_plate_spline import ThinPlateSpline


class PatchProcessorAndApplier(Module):
    def __init__(
        self,
        patch_transformer: PatchTransformer,
        patch_crop: Optional[PatchCropping] = None,
        tps_sampler: Optional[ThinPlateSpline] = None,
    ) -> None:
        super().__init__()
        self.patch_transformer = patch_transformer
        self.patch_applier = PatchApplier()
        self.patch_crop = patch_crop
        self.tps_sampler = tps_sampler

    def forward(self, patch: Tensor, image: Tensor, target: Tensor, target_len: Tensor):
        if self.patch_crop is not None:
            patch = self.patch_crop.forward(patch)
        if self.tps_sampler is not None:
            patch = self.tps_sampler.forward(patch)
        adv_batch, msk_batch = self.patch_transformer.forward(patch, target)
        return self.patch_applier.forward(image, target_len, adv_batch, msk_batch)
