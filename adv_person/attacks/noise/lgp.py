"""Simple noise attack using Adamax optimizer for 50 iterations."""

import math
from functools import partial
from numbers import Number
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import Adamax

from adv_person.models import IDetector
from adv_person.utils.bbox_utils import bbox_ious

from .adversary import Adversary


def generate_object_heatmap(X: Tensor, gt_bboxes: Tensor, delta=1.5):
    """
    Args:
        img_shape: (C, H, W) input image shape
        gt_bboxes: (M,4/5) GT bboxes (xyxy/xywhr)
        delta: scaling ratio for perturbed space (Paper's δ=1.5)
        device: cuda/cpu
    Returns:
        heatmap: (H, W) object-wise weight map (0~1, foreground=low weight, background=1)
    """
    device = X.device
    H, W = X.shape[-2:]
    heatmap = torch.ones((H, W), device=device)  # background default=1
    for gt in gt_bboxes:
        # Get bbox center (xc, yc) and size (w, h) (compatible with rotated boxes)
        x1, y1, x2, y2 = gt[:4] * gt.new_tensor((W, H, W, H))
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        # Expand perturbed space by delta (Paper's δB^gt)
        w_exp, h_exp = w * delta, h * delta
        # Generate grid for Euclidean distance weight
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device)
        )
        dist = (
            0.5
            * torch.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)
            / torch.sqrt(w_exp**2 + h_exp**2)
        )
        # Gaussian-like weight (Eq.(7))
        fg_weight = torch.clamp(dist, 0, 1)
        heatmap = torch.min(heatmap, fg_weight)  # overlap GTs take min weight
    return heatmap.unsqueeze(0)  # (1,H,W) for broadcast


class LGP(Adversary):
    """Simple noise attack using Adamax optimizer.

    Attributes:
        eps (float):
            Attack epsilon.
        p (float | int):
            Bound the perturbation by L_p norm.
        iterations (int):
            Number of optimization iterations.
        lr (float):
            Learning rate for Adamax optimizer.
        use_amp (bool):
            Whether to use Automatic Mixed Precision (AMP).
            Default: False.
        amp_autocast (function () -> context):
            The AMP autocast function that create the autocast context.
            Default: torch autocast to CUDA device.
        grad_scaler (amp.GradScaler | None):
            Gradient scaler for AMP. Default: None.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        p: Number = math.inf,
        iterations: int = 50,
        lr: float = 0.1,
        weight_decay: float = 0.02,
        iou_thresh: float = 0.01,
        lambda_cls=1.0,
        iou_topk=5,
        cls_topk=5,
        use_amp=False,
        amp_autocast: Optional[partial[torch.autocast]] = None,
        grad_scaler=None,
    ):
        super().__init__(
            eps=eps,
            p=p,
            use_amp=use_amp,
            amp_autocast=amp_autocast,
            grad_scaler=grad_scaler,
        )
        self.iterations = iterations
        self.lr = lr
        self.weight_decay = weight_decay
        self.iou_thresh = iou_thresh
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")
        self.lambda_cls = lambda_cls
        self.iou_topk = iou_topk
        self.cls_topk = cls_topk
        self.target_cls = 0  # person class
        self.normalize = None

    def _get_boxes(self, model: Module, data: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Get boxes from model using the correct inference interface.

        Args:
            model (Module): Model to get boxes from.
            data (Tensor): Input tensor.

        Returns:
            list[tuple[Tensor, Tensor]]: List of (boxes, labels) tuples.
        """
        from adv_person.utils.bbox_utils import bbox_to_x1y1x2y2

        # Get model name if available
        model_name = getattr(model, "name", None)
        if self.normalize is not None:
            data = self.normalize(data)

        if hasattr(model, "forward_test"):
            # Use forward_test for models that have it
            if model_name in ("YOLOv2", "YOLOv9"):
                return model.forward_test(
                    data, conf_thresh=0.01 if model_name == "YOLOv2" else 0.001
                )
            else:
                return model.forward_test(data)
        elif isinstance(model, IDetector):
            # Use forward_test for IDetector implementations
            return model.forward_test(data)
        else:
            # Fallback for other models
            raise NotImplementedError
            results = model(data)
            all_boxes = []

            # Handle different result formats
            if isinstance(results, list):
                # Assume list of dictionaries (like TorchVision detection models)
                for result in results:
                    if isinstance(result, dict):
                        boxes = result["boxes"]
                        scores = result["scores"]
                        labels = result["labels"]
                        # Normalize boxes to [0,1]
                        H, W = data.shape[2:]
                        boxes = boxes / boxes.new_tensor((W, H, W, H))
                        all_boxes.append(
                            (torch.cat([boxes, scores.unsqueeze(1)], dim=1), labels)
                        )
            elif isinstance(results, dict) and "pred_boxes" in results:
                # Assume DETR-like output
                raw_boxes = results["pred_boxes"]
                logits_name = "pred_logits" if "pred_logits" in results else "logits"
                all_scores, all_labels = torch.max(
                    results[logits_name].softmax(dim=-1), dim=-1
                )
                for i, boxes in enumerate(raw_boxes):
                    boxes = bbox_to_x1y1x2y2(boxes, "cxcywh")
                    scores = all_scores[i]
                    labels = all_labels[i]
                    all_boxes.append(
                        (torch.cat([boxes, scores.unsqueeze(1)], dim=1), labels)
                    )

            return all_boxes

    def _attack(
        self, model: Module, criterion: Callable, X: Tensor, y: Tensor
    ) -> Tensor:
        """Core function for attack implementation.

        Args:
            model (Module):
                Full model with normalization to attack.
            criterion (Callable):
                Loss function to attack (ignored, using hardcoded loss).
            X ((N,C,H,W) Tensor):
                Input batch.
            y ((N,) Tensor | None):
                Batch of y_true in untargeted attack or target in targeted attack.

        Returns:
            X_adv ((N,C,H,W) Tensor):
                Adversarial input batch.
        """
        # Initialize adversarial image
        delta = torch.zeros_like(X).requires_grad_(True)
        ZERO = X.new_zeros(())
        heatmap = torch.stack(
            [generate_object_heatmap(X, target) for target in y], dim=0
        )

        # Initialize optimizer
        optimizer = Adamax([delta], lr=self.lr, weight_decay=self.weight_decay)

        # Perform optimization for specified iterations
        for _ in range(self.iterations):
            optimizer.zero_grad()
            X_adv = X + delta
            boxes_with_labels = self._get_boxes(model, X_adv)
            losses: list[Tensor] = []
            for (boxes, labels), targets in zip(boxes_with_labels, y):
                if len(boxes) > 0 and len(targets) > 0:
                    # Compute IoU between detections and targets
                    iou = bbox_ious(boxes, targets)
                    iou_max: Tensor = iou.max(dim=1)[0]
                    iou_ind = iou_max.sort(descending=True).indices[: self.iou_topk]
                    iou_boxes = boxes[iou_ind][:, :4]
                    iou_loss = self.smooth_l1(iou_boxes, torch.zeros_like(iou_boxes))

                    # valid_cls_ind = (iou_max > self.iou_thresh) & (
                    #     labels == self.target_cls
                    # )
                    # valid_cls_boxes = boxes[valid_cls_ind]
                    valid_cls_boxes = boxes
                    # cls_ind = valid_cls_boxes[:, 4].sort(descending=True).indices[:self.cls_topk]
                    # unique_ind = torch.unique(torch.cat([iou_ind, cls_ind], dim=0))
                    cls_loss = (
                        valid_cls_boxes[:, 4]
                        .sort(descending=True)
                        .values[: self.cls_topk]
                        .mean()
                    )
                    percept_loss = self.smooth_l1(heatmap * delta, ZERO)
                else:
                    iou_loss = boxes.new_zeros((), requires_grad=boxes.requires_grad)
                    cls_loss = boxes.new_zeros((), requires_grad=boxes.requires_grad)
                    percept_loss = boxes.new_zeros(
                        (), requires_grad=boxes.requires_grad
                    )

                losses.append(
                    iou_loss + self.lambda_cls * cls_loss + 0.1 * percept_loss
                )
            loss = torch.stack(losses).sum()

            # Backward pass
            if self.use_amp and self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Project back to epsilon ball and valid image range
            with torch.no_grad():
                # Calculate perturbation
                # Project to L_p ball
                # if self.p == math.inf:
                #     delta.clamp_(-self.eps, self.eps)
                # else:
                #     norm = delta.view(delta.size(0), -1).norm(p=self.p, dim=1)
                #     scale = torch.clamp_max(norm / self.eps, 1.0)
                #     delta /= scale.view(delta.size(0), 1, 1, 1)
                delta.clamp_(0 - X, 1 - X)

        with torch.no_grad():
            # norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
            # print("L2 norms:", norm)
            delta.clamp_(-self.eps, self.eps)
            X_adv = (X + delta).clamp_(0, 1)
        return X_adv.detach()
