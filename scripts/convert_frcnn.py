import re

import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

ckpt = torch.load("data/frcnn_r50_pgdat.pth", map_location="cpu")
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

head_maps = {
    "rpn_head.rpn_conv": "rpn.head.conv.0.0",
    "rpn_head.rpn_cls": "rpn.head.cls_logits",
    "rpn_head.rpn_reg": "rpn.head.bbox_pred",
    "roi_head.bbox_head.shared_fcs.0": "roi_heads.box_head.fc6",
    "roi_head.bbox_head.shared_fcs.1": "roi_heads.box_head.fc7",
    "roi_head.bbox_head.fc_cls": "roi_heads.box_predictor.cls_score",
    "roi_head.bbox_head.fc_reg": "roi_heads.box_predictor.bbox_pred",
}

model = fasterrcnn_resnet50_fpn(num_classes=81)
model_state_dict = model.state_dict()

state_dict_new: dict[str, torch.Tensor] = {}
for key, value in ckpt["state_dict"].items():
    if key.startswith("backbone"):
        if key.endswith("num_batches_tracked"):
            continue
        key = key.replace("backbone.", "backbone.body.")
    elif key.startswith("neck"):
        key = re.sub(
            r"neck\.lateral_convs\.([0-9]).conv", r"backbone.fpn.inner_blocks.\1.0", key
        )
        key = re.sub(
            r"neck\.fpn_convs\.([0-9]).conv", r"backbone.fpn.layer_blocks.\1.0", key
        )
    else:
        for k, v in head_maps.items():
            key = key.replace(k, v)
    if "roi_heads.box_predictor.bbox_pred" in key:
        t = torch.zeros_like(model_state_dict[key])
        t[4:] = value
        state_dict_new[key] = t
    elif "roi_heads.box_predictor.cls_score" in key:
        t = torch.zeros_like(model_state_dict[key])
        t[0] = value[-1]
        t[1:] = value[:-1]
        state_dict_new[key] = t
    else:
        state_dict_new[key] = value

assert set(state_dict_new.keys()) == set(model_state_dict.keys())

shape_mismatch = False
for key, value in model_state_dict.items():
    if value.shape != state_dict_new[key].shape:
        print(f"Shape mismatch: {key}: {value.shape} vs {state_dict_new[key].shape}")
        shape_mismatch = True

ckpt_dst = "faster_rcnn_r50_fpn_at.pt"
if not shape_mismatch:
    torch.save(state_dict_new, ckpt_dst)
    print("Checkpoint saved:", ckpt_dst)
