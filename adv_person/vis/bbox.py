from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

from adv_person.utils.bbox_utils import bbox_scale


def get_pil_with_bbox(
    image: Tensor,
    boxes: Tensor,
    scores: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    label_map: Optional[list[str]] = None,
    **kwargs,
) -> Image.Image:
    _, h, w = image.shape
    args = {}
    args["image"] = TF.convert_image_dtype(image, torch.uint8)
    args["boxes"] = bbox_scale(boxes, (w, h))
    if labels is not None:
        if label_map is None:
            text_labels = [str(l) for l in labels.tolist()]
        else:
            text_labels = [label_map[l] for l in labels.tolist()]
        args["labels"] = text_labels
    if scores is not None:
        text_scores = ["{:.2f}".format(x) for x in scores.tolist()]
        if "labels" in args:
            args["labels"] = [" ".join(l) for l in zip(args["labels"], text_scores)]
        else:
            args["labels"] = text_scores
    args.update(kwargs)
    image_with_box = draw_bounding_boxes(**args)
    return TF.to_pil_image(image_with_box)
