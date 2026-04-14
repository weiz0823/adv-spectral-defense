from typing import Optional

import torch
from torch import Tensor, optim
from torch.nn import Module
from tqdm import tqdm

from adv_person.models.i_detector import IDetector
from adv_person.models.loss import DetectionLoss, total_variation

from .patch.patch_processor import PatchProcessorAndApplier


class SamplewisePatchAttack(object):
    def __init__(
        self, lr=0.03, steps=600, patch_init=0.5, patch_size=300, tv_loss=2.5
    ) -> None:
        self.lr = lr
        self.steps = steps
        self.patch_init = patch_init
        self.patch_size = patch_size
        self.tv_loss = tv_loss

    @staticmethod
    def init_patch(patch: Tensor, patch_init):
        if isinstance(patch_init, float):
            patch.fill_(patch_init)
        elif isinstance(patch_init, int):
            patch.fill_(patch_init / 255)
        elif isinstance(patch_init, str):
            if patch_init == "random":
                patch.uniform_()
            else:
                raise ValueError(patch_init)
        else:
            raise ValueError(patch_init)

    def attack(
        self,
        model: Module,
        criterion: DetectionLoss,
        patch_applier: PatchProcessorAndApplier,
        data: Tensor,
        target: Tensor,
        target_lens: Tensor,
        tqdm_obj: Optional[tqdm] = None,
    ):
        assert isinstance(model, IDetector)
        model.eval()
        patch = torch.empty(
            (len(target), 3, self.patch_size, self.patch_size), device=data.device
        )
        self.init_patch(patch, self.patch_init)
        patch.requires_grad_()
        optimizer = optim.Adam([patch], lr=self.lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=50, cooldown=500, min_lr=self.lr / 100
        )
        use_tv_loss = self.tv_loss > 0.0
        target_list = target.split(list(target_lens))
        for step in range(self.steps):
            if tqdm_obj is not None:
                tqdm_obj.update()
            patch.detach_().requires_grad_()
            data.detach_()
            target.detach_()
            target_lens.detach_()
            optimizer.zero_grad(True)
            patched_data = patch_applier.forward(patch, data, target, target_lens)
            all_boxes: list[tuple[Tensor, Tensor]] = model.forward_test(patched_data)
            boxes = [box for box, label in all_boxes]
            det_loss = criterion.forward(boxes, target_list)
            loss = det_loss
            if use_tv_loss:
                tv_loss = torch.clamp_min(self.tv_loss * total_variation(patch), 0.1)
                loss = loss + tv_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)  # Keep patch in image range
            if step > 100:
                scheduler.step(loss.item())
        return patch
