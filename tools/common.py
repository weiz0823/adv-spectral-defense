import logging
import os
import os.path as osp
from logging import Logger
from typing import Callable, Optional

import torch
import torchvision
from torch import Tensor, nn, optim
from torch.nn import Module
from torchvision import transforms

from adv_person import configs
from adv_person.configs import CLIConfig
from asd_defense import DWTPreprocessPlugin
from adv_person.models import *
from adv_person.utils.bbox_utils import bbox_to_x1y1x2y2

PREPROCESS_DEFENSES = [
    "DWT",
    "none",
    "FFT",
    "DCT",
]


def get_obj_cls_loss_fn(arg: str) -> Callable[[Tensor, Tensor], Tensor]:
    if arg == "obj_cls":
        obj_cls_loss = lambda obj, cls: obj * cls
    elif arg == "obj":
        obj_cls_loss = lambda obj, cls: obj
    elif arg == "cls":
        obj_cls_loss = lambda obj, cls: cls
    else:
        raise ValueError(arg)
    return obj_cls_loss


_mmdet_model_dict = {
    "YOLOv3": (
        "YOLOV3Detector",
        "configs/yolo/yolov3_d53_mstrain-416_273e_coco.py",
        "yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
    ),
    "FasterRCNN": (
        "FasterRCNNDetector",
        "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
        "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    ),
    "MaskRCNN": (
        "MaskRCNNDetector",
        "configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
        "mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
    ),
    "CascadeRCNN": (
        "CascadeRCNNDetector",
        "configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py",
        "cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth",
    ),
    "FCOS": (
        "FCOSDetector",
        "configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py",
        "fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
    ),
    "FCOS-AT": (
        "FCOSDetector",
        "fcos-configs/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py",
        "fcos_r50_pgdat.pth",
    ),
    "RetinaNet": (
        "RetinaNetDetector",
        "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
        "retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
    ),
    "DeformableDETR": (
        "DeformableDETRDetector",
        "configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
        "deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth",
    ),
}


def load_clean_model_mmdet(device: torch.device, args: CLIConfig) -> Module:
    from mmdet.apis import init_detector

    mmdet_root = osp.expanduser("../mmdetection")
    model_type, cfg_file, default_weight = _mmdet_model_dict[args.model]
    if args.weight.endswith((".pth", ".pt")):
        weight = args.weight
    else:
        weight = osp.join(args.weight, default_weight)
    cfg_options = dict(model=dict(type=model_type))
    if args.model in ("FCOS", "FCOS-AT"):
        cfg_options["model"]["bbox_head"] = dict(type="FCOSHeadExtend")
    model = init_detector(
        osp.join(mmdet_root, cfg_file),
        weight,
        device=device,
        cfg_options=cfg_options,
    )
    return model


def load_clean_model(device: torch.device, args: CLIConfig) -> Module:
    obj_cls_loss = get_obj_cls_loss_fn(args.obj_cls_loss)

    if args.model in _mmdet_model_dict:
        model = load_clean_model_mmdet(device, args)
    elif args.model == "YOLOv2":
        model = YOLOV2(
            conf_thresh=args.conf_thresh,
            weightfile=args.weight,
            obj_cls_loss=obj_cls_loss,
        )
    elif args.model == "YOLOv9":
        from adv_person.models.yolov9_detector import YOLOv9Detector

        model = YOLOv9Detector(args.weight, device, fuse=False)
    elif args.model == "FasterRCNN_torch":
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        ).to(device)
    elif args.model == "FasterRCNN_AT":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            num_classes=81, min_size=416
        )
        model.load_state_dict(torch.load(args.weight, map_location="cpu"))
        model = model.to(device)
    elif args.model == "MaskRCNN_torch":
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        ).to(device)
    elif args.model == "DETR":
        model = torch.hub.load(
            "facebookresearch/detr:main", "detr_resnet50", pretrained=True
        ).to(device)
    elif args.model == "Deformable-DETR":
        from transformers import DeformableDetrForObjectDetection

        model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr"
        ).to(device)
    elif args.model == "RF-DETR":
        from adv_person.models.rfdetr import RFDETRDetector

        model = RFDETRDetector(
            type_="RFDETRSmall", pretrain_weights="data/rf-detr-small.pth"
        )
    else:
        raise ValueError(args.model)
    return model


def load_model(device: torch.device, args: CLIConfig) -> Module:
    if args.defense in PREPROCESS_DEFENSES:
        return load_clean_model(device, args)
    else:
        # Load models with defense applied
        raise NotImplementedError


def get_target_cls(args: CLIConfig):
    if args.model in (
        "FasterRCNN_torch",
        "FasterRCNN_AT",
        "MaskRCNN_torch",
        "Deformable-DETR",
        "RF-DETR",
    ):
        target_cls = 1
    else:
        target_cls = 0
    return target_cls


def get_normalization(args: CLIConfig):
    if args.model in (
        "FasterRCNN",
        "MaskRCNN",
        "CascadeRCNN",
        "RetinaNet",
        "FCOS-AT",
        "DeformableDETR",
    ):
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)
        return transforms.Normalize([x / 255 for x in mean], [x / 255 for x in std])
    elif args.model == "FCOS":
        mean = (102.9801, 115.9465, 122.7717)
        std = (1, 1, 1)
        return transforms.Normalize([x / 255 for x in mean], [x / 255 for x in std])
    elif args.model == "Deformable-DETR":
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        # YOLO series do not need normalization
        return nn.Identity()


def get_defense_preprocess(args: CLIConfig):
    if args.defense == "DWT":
        dwt_pool = dict(
            type="AvgPool2d",
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return DWTPreprocessPlugin(
            args.dwt_max_level,
            args.dwt_thresh,
            args.dwt_thresh_multiplier,
            args.dwt_process,
            args.dwt_kern_size,
            dwt_pool,
        )
    elif args.defense == "FFT":
        from adv_person.defense.freq import FFTPreprocess

        return FFTPreprocess(
            threshold=getattr(args, "fft_threshold", 0.1),
            high_pass=getattr(args, "fft_high_pass", False),
            keep_ratio=getattr(args, "fft_keep_ratio", 0.8),
        )
    elif args.defense == "DCT":
        from adv_person.defense.freq import DCTPreprocess

        return DCTPreprocess(
            threshold=getattr(args, "dct_threshold", 0.1),
            high_pass=getattr(args, "dct_high_pass", False),
            keep_ratio=getattr(args, "dct_keep_ratio", 0.8),
        )
    else:
        return nn.Identity()


def apply_defense_preprocess(
    module: Module, input: Tensor, target: Optional[Tensor] = None, **kwargs
) -> tuple[Tensor, Tensor] | Tensor:
    rv = module(input, **kwargs), target
    if target is None:
        return rv[0]
    else:
        return rv


def get_boxes(model: Module, data: Tensor, args: CLIConfig):
    if args.model == "YOLOv2":
        all_boxes: list[tuple[Tensor, Tensor]] = model.forward_test(
            data, conf_thresh=0.01
        )
    elif args.model in ("MaskRCNN_torch", "FasterRCNN_torch", "FasterRCNN_AT"):
        results = model(data)
        all_boxes: list[tuple[Tensor, Tensor]] = []
        H, W = data.shape[2:]
        for result in results:
            boxes: Tensor = result["boxes"]
            scores: Tensor = result["scores"]
            labels: Tensor = result["labels"]
            boxes = boxes / boxes.new_tensor((W, H, W, H))
            all_boxes.append((torch.cat([boxes, scores.unsqueeze(1)], dim=1), labels))
    elif args.model in ("DETR", "Deformable-DETR"):
        logits_name = "pred_logits" if args.model == "DETR" else "logits"
        results = model(data)
        all_boxes: list[tuple[Tensor, Tensor]] = []
        raw_boxes = results["pred_boxes"]
        all_scores, all_labels = torch.max(results[logits_name].softmax(dim=-1), dim=-1)
        for i, boxes in enumerate(raw_boxes):
            boxes = bbox_to_x1y1x2y2(boxes, "cxcywh")
            scores = all_scores[i]
            labels = all_labels[i]
            all_boxes.append((torch.cat([boxes, scores.unsqueeze(1)], dim=1), labels))
    elif args.model in ("YOLOv9",):
        all_boxes: list[tuple[Tensor, Tensor]] = model.forward_test(
            data, conf=args.conf_thresh
        )
    else:
        assert isinstance(model, IDetector)
        all_boxes: list[tuple[Tensor, Tensor]] = model.forward_test(data)
    return all_boxes


def average(x):
    return sum(x) / len(x)


def log_contents(logger: Optional[Logger], lines: list):
    for line in lines:
        if logger is not None:
            logger.info(line)
        else:
            print(line)
