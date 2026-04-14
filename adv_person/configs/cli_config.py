from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from typing import Optional

from .custom_types import int_or_float, user_path


class CLIConfig(Namespace):
    """Namespace for attribute and type hint on CLI configurations.
    You can also specify default values here."""

    attack: str
    attack_variant: str
    angle: tuple[float, float]
    brightness: tuple[float, float]
    batch_size: int
    batch_size_test: int
    clamp_shift: float
    color_transform: bool
    conf_thresh: float
    contrast: tuple[float, float]
    ckpt_interval: int
    crop: str
    crop_size: Optional[int]
    data_dir: str
    defense: str
    dwt_kern_size: int
    dwt_max_level: int
    dwt_extra_levels: list[int]
    dwt_process: str
    dwt_thresh: float
    dwt_thresh_multiplier: int | float
    epochs: int
    eps: float
    eval: bool
    eval_best: bool
    eval_clean: bool
    expr_suffix: Optional[str]
    img_size: int
    iou_thresh: float
    iou_thresh_test: float
    label_dir: Optional[str]
    lc_scale: float
    log: list[str]
    lr: float
    lr2: float
    model: str
    nms_thresh: float
    noise: float
    nps_color_file: str
    nps_loss: float
    obj_cls_loss: str
    old_fashion: bool
    output_dir: str
    overwrite: bool
    patch: str
    patch_init: str | float
    patch_size: int
    pooling: str
    ratio: float
    resume: int
    scale: float
    scale_range: tuple[float, float]
    seed: Optional[int]
    subset: int
    tensorboard_interval: int
    test_repeats: int
    texture: str
    transform_fixed: bool
    translation: tuple[float, float]
    tps: bool
    tps3d: bool
    tv_loss: float
    weight: str
    workers: int


def get_parser():
    """All arguments which can be specified from command line."""
    parser = ArgumentParser()
    parser.add_argument(
        "--angle",
        type=float,
        nargs=2,
        default=(-20.0, 20.0),
        help="Patch rotation angle range",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="patch",
        help="Attack type",
    )
    parser.add_argument(
        "--attack_variant",
        type=str,
        default="default",
        help="Attack variant",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size in training",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=16,
        help="Batch size in testing",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        nargs=2,
        default=(-0.1, 0.1),
        help="Patch brightness range",
    )
    parser.add_argument(
        "--clamp_shift",
        type=float,
        default=0.0,
        help="Shift of clamping. clamp(0, 1) -> clamp(clamp_shift, 1-clamp_shift).",
    )
    parser.add_argument(
        "--color_transform",
        action=BooleanOptionalAction,
        default=True,
        help="Enable/disable color transformation, useful in camouflage attack.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.25,
        help="Confidence score threshold",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        nargs=2,
        default=(0.8, 1.2),
        help="Patch contrast range",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=200,
        help="Checkpoint interval (by epoch)",
    )
    parser.add_argument(
        "--crop",
        choices=["none", "TCA", "jitter"],
        default="none",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        help="Crop size in cropping attacks",
    )
    parser.add_argument(
        "--data_dir",
        type=user_path,
        default=user_path("data/INRIAPerson"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        help="Defense to use",
    )
    # DWT configs
    parser.add_argument(
        "--dwt_kern_size",
        type=int,
        default=33,
        help="DWT blurring kernel size",
    )
    parser.add_argument(
        "--dwt_max_level",
        type=int,
        default=4,
        help="Maximum expansion level of DWT",
    )
    parser.add_argument(
        "--dwt_extra_levels",
        type=int,
        nargs="+",
        default=[],
        help="Extra aggregated levels of DWT",
    )
    parser.add_argument(
        "--dwt_process",
        choices=["blur", "mask"],
        help="DWT preprocess type",
    )
    parser.add_argument(
        "--dwt_thresh",
        type=float,
        default=4,
        help="DWT clipping threshold",
    )
    parser.add_argument(
        "--dwt_thresh_multiplier",
        type=int_or_float,
        default=2,
        help=(
            "DWT clipping threshold multiplier M, and actual threshold for"
            " level k = M**k"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Total epochs to run",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluation mode",
    )

    # Patch loading mutually exclusive group
    group = parser  # .add_mutually_exclusive_group()
    group.add_argument(
        "--eval_best",
        action="store_true",
        help="Evaluate best checkpoint",
    )
    group.add_argument(
        "--eval_clean",
        action="store_true",
        help="Evaluate on clean samples",
    )
    group.add_argument(
        "--resume",
        type=int,
        default=-1,
        help="Resume specific epoch",
    )
    group.add_argument(
        "--patch",
        type=user_path,
        default="",
        help="Patch file to load",
    )

    parser.add_argument(
        "--expr_suffix",
        "--suffix",
        help="Experiment suffix",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=416,
        help="Image size",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.2,
        help="IoU threshold in training",
    )
    parser.add_argument(
        "--iou_thresh_test",
        type=float,
        default=0.5,
        help="IoU threshold for mAP calculation",
    )
    parser.add_argument(
        "--label_dir",
        type=lambda x: user_path(x) if x is not None else x,
        help="Custom label directory",
    )
    parser.add_argument(
        "--lc_scale",
        type=float,
        default=0.1,
        help="Random location scale for patch",
    )
    parser.add_argument(
        "--log",
        nargs="+",
        choices=["file", "tensorboard", "csv", "stream"],
        default=["tensorboard", "file"],
        help="Log outputs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr2",
        type=float,
        default=0.03,
        help="Learning rate of second optimizer",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="YOLOv2",
        help="Model to attack",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.4,
        help="IoU threshold used in Non-Maximum Suppression",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Noise factor for patch",
    )
    parser.add_argument(
        "--nps_color_file",
        type=user_path,
        default="configs/nps_30values.txt",
        help="NPS color file",
    )
    parser.add_argument(
        "--nps_loss",
        type=float,
        default=0.0,
        help="NPS loss coefficient",
    )
    parser.add_argument(
        "--obj_cls_loss",
        choices=["obj_cls", "obj", "cls"],
        default="obj_cls",
        help="Object or classification loss",
    )
    parser.add_argument(
        "--old_fashion",
        action=BooleanOptionalAction,
        default=False,
        help="old fashion in patch applier",
    )
    parser.add_argument(
        "--output_dir",
        type=user_path,
        default=user_path("./output"),
        help="Output root directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing training directory",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=300,
        help="Patch size",
    )
    parser.add_argument(
        "--pooling",
        default="median",
        help="Pooling",
    )
    parser.add_argument(
        "--ratio",
        "--seed_ratio",
        type=float,
        default=1.0,
        help="Patch y/x ratio. [Shared] Seed ratio in camouflage attack.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.2,
        help="Patch scale",
    )
    parser.add_argument(
        "--scale_range",
        type=float,
        nargs=2,
        default=(0.75, 1.6),
        help="Patch scale range",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set random seed",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=-1,
        help="Subset size",
    )
    parser.add_argument(
        "--tensorboard_interval",
        type=int,
        default=5,
        help="TensorBoard log interval (by iteration)",
    )
    parser.add_argument(
        "--test_repeats",
        type=int,
        default=5,
        help="Test repeats",
    )
    parser.add_argument(
        "--texture",
        choices=["simple", "camouflage"],
        default="camouflage",
    )
    parser.add_argument(
        "--transform_fixed",
        action=BooleanOptionalAction,
        default=False,
        help="Enable/disable fixed transformation",
    )
    parser.add_argument(
        "--translation",
        type=float,
        nargs=2,
        default=(0.8, 1.0),
        help="XY translation magnitude",
    )
    parser.add_argument(
        "--tps",
        action=BooleanOptionalAction,
        default=False,
        help="Enable Thin Plate Spline sampling",
    )
    parser.add_argument(
        "--tps3d",
        action=BooleanOptionalAction,
        default=False,
        help="Enable Thin Plate Spline sampling on 3D model",
    )
    parser.add_argument(
        "--tv_loss",
        type=float,
        default=0.0,
        help="TV loss coefficient",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--weight",
        metavar="FILE",
        default="data/yolov2.weights",
        help="Model pretrained weight",
    )
    parser.add_argument("--comment", "-m", default="This is the default comment.")

    return parser
