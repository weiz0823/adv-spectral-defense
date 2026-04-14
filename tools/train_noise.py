"""Train noise-like attack using NumbODAdversary on INRIA Person dataset."""

import logging
import os
import os.path as osp
import random
import sys
import time
from typing import Callable, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tensorboardX import SummaryWriter
from torch import Tensor, nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import batched_nms, nms
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))
from adv_person import configs, utils
from adv_person.attacks.noise import Adversary, NumbOD, LGP, SimpleNoiseAdversary
from adv_person.configs import CLIConfig
from adv_person.datasets import InriaDataset, detection_collate
from adv_person.defense import EPGFPreprocess
from adv_person.models import *
from adv_person.models.loss import DetectionLoss
from adv_person.utils.meter import *
from adv_person.utils.stats_meter import *
from adv_person.utils.torch_utils import gen_uniform_grid
from tools.common import *
from tools.expr_path import *

try:
    from adv_person.models.mmdet import *
except ImportError:
    logging.warning("MMDetection models unavailable!")


@torch.no_grad()
def test_ap(
    model: Module,
    dataloader: DataLoader,
    adversary: Optional[Adversary],
    device: torch.device,
    args: configs.CLIConfig,
    tqdm_obj: Optional[tqdm] = None,
    preprocess: Optional[Module] = None,
):
    model.eval()
    criterion = DetectionLoss(
        args.iou_thresh_test,
        get_obj_cls_loss_fn(args.obj_cls_loss),
        reduction="mean_test",
    )
    if preprocess is None:
        preprocess = get_defense_preprocess(args).to(device)
    if args.defense == "NAPGuard":
        preprocess.eval()
    target_cls = get_target_cls(args)
    normalize = get_normalization(args).to(device)

    map_meter = MeanAveragePrecision(
        iou_thresholds=[args.iou_thresh_test],
        max_detection_thresholds=[10, 100, 1000],
    )
    asr_meter = utils.AttackSuccessRate(iou_threshold=args.iou_thresh_test)
    test_time = SimpleMeter("test_time", "{:.3f}")
    test_loss_det = AverageMeter("test_loss_det", "{:.3f}")
    meters: list[Meter] = list(
        utils.filter_not_none(
            [
                test_time,
                test_loss_det,
            ]
        )
    )

    begin_time = time.time()
    from torchvision.ops import nms

    for batch_idx, (data, target, target_lens) in enumerate(dataloader):
        data: Tensor
        target: Tensor
        target_lens: Tensor
        cumsum_lens = torch.cumsum(target_lens, 0)
        data = data.to(device)
        target = target.to(device)

        # Generate adversarial images if adversary is provided
        if adversary is not None:
            adversary.target_cls = target_cls
            adversary.normalize = normalize
            data_adv = adversary.attack(
                model, criterion, data, target.split(list(target_lens))
            )
            data = data_adv

        # if batch_idx == 0:
        #     os.makedirs("frames", exist_ok=True)
        # data = data.cpu()
        # for i, img in enumerate(data):
        #     img_name = dataloader.dataset.img_names[batch_idx * args.batch_size_test + i]
        #     save_image(img, f"frames/{img_name}")
        # continue
        # breakpoint()

        if isinstance(preprocess, DWTPreprocessPlugin):
            # Targets not modified in this case
            preprocessed_input, _ = apply_defense_preprocess(preprocess, data, target)
            level_boxes = []
            boxes = get_boxes(model, normalize(preprocessed_input), args)
            level_boxes.append(boxes)
            extra_levels = args.dwt_extra_levels
            if len(extra_levels) > 0:
                # Forward multiple levels
                for level in extra_levels:
                    preprocess.level_override = level
                    preprocessed_input, _ = apply_defense_preprocess(
                        preprocess, data, target
                    )
                    boxes = get_boxes(model, normalize(preprocessed_input), args)
                    level_boxes.append(boxes)
                preprocess.level_override = -1
                # Aggregate multiple levels
                all_boxes: list[tuple[Tensor, Tensor]] = []
                for j in range(len(level_boxes[0])):
                    boxes = torch.cat([b[j][0] for b in level_boxes], 0)
                    labels = torch.cat([b[j][1] for b in level_boxes], 0)
                    keep = batched_nms(
                        boxes[..., :4], boxes[..., 4], labels, args.nms_thresh
                    )
                    all_boxes.append((boxes[keep], labels[keep]))
            else:
                all_boxes = level_boxes[0]
        else:
            data, target = apply_defense_preprocess(preprocess, data, target)
            all_boxes: list[tuple[Tensor, Tensor]] = get_boxes(
                model, normalize(data), args
            )
        # save_image(data[0], "tmp_img.jpg")
        # breakpoint()
        det_loss = criterion.forward(
            [box for box, idx in all_boxes], target.split(list(target_lens))
        )
        test_loss_det.update(det_loss.item(), len(data))
        for i, (boxes_with_score, cls_idx) in enumerate(all_boxes):
            boxes = boxes_with_score[:, :4]
            scores = boxes_with_score[:, 4]
            # Filter with NMS
            kept = nms(boxes, scores, args.nms_thresh)
            # Filter person class
            kept = kept[cls_idx[kept] == target_cls]
            boxes = boxes[kept]
            scores = scores[kept]
            cls_idx = cls_idx[kept]

            end = cumsum_lens[i].item()
            count = target_lens[i].item()
            local_target = target[end - count : end]
            target_labels = cls_idx.new_full((len(local_target),), target_cls)
            # The original metric requires absolute box coordinates,
            # but I don't think it would be different for relative ones
            preds = [dict(boxes=boxes, scores=scores, labels=cls_idx)]
            tgts = [dict(boxes=local_target, labels=target_labels)]
            asr_meter.update(preds, tgts)
            map_meter.update(preds, tgts)
            # DEBUG
            # sample_idx = batch_idx * args.batch_size_test + i
            # current_asr = (asr_meter.all_conf_scores[-1] <= 0.5).float().mean().item()
            # success_count = int(current_asr * len(local_target))
            # print(
            #     f"[{sample_idx}/{len(dataloader.dataset)}] ASR: {success_count}/{len(local_target)}"
            # )
        if tqdm_obj is not None:
            tqdm_obj.update()
    test_time.update(time.time() - begin_time)

    # print("Finish saving images!")
    # exit()

    return meters, map_meter, asr_meter


def main(args: CLIConfig = None):
    if args is None:
        args = configs.get_args()
    assert args.attack == "noise", "Noise attacks should use `train_noise.py`"
    if args.eval_best or args.eval_clean:
        args.eval = True
    assert args.eval or args.resume < 0, "Resume on training to be implemented"

    # Here are some coupled arguments set for convenience
    if args.eval and "tensorboard" in args.log:
        args.log = ["stream"]
    args.test_repeats = 1

    is_train = not args.eval
    is_eval = args.eval
    is_new_path = True
    device = utils.get_device()
    print("Device is", device, file=sys.stderr)
    expr_path = get_expr_path(args, train=is_new_path)
    print(f"Experiment path is '{expr_path}'", file=sys.stderr)
    if is_new_path and "file" in args.log:
        if not args.overwrite and osp.exists(expr_path):
            print(
                "Experiment path exists, please specify '--overwrite' to force overwrite",
                file=sys.stderr,
            )
            return
        os.makedirs(expr_path, exist_ok=True)

    # Random seed
    utils.seed_rngs(args.seed)

    # Setup logging
    if "tensorboard" in args.log and is_train:
        from adv_person.utils.tensorboard import init_tensorboard

        summary_writer = init_tensorboard(
            osp.join(args.output_dir, "runs"), get_tensorboard_name(args, expr_path)
        )
        print(f"Tensorboard log dir is '{summary_writer.logdir}'", file=sys.stderr)
    else:
        summary_writer = None
    log_filename = "train.log" if is_train else "eval.log"
    log_file = osp.join(expr_path, log_filename) if "file" in args.log else None
    log_stream = sys.stderr if "stream" in args.log else None
    csv_filename = "train.csv" if is_train else "eval.csv"
    csv_file = osp.join(expr_path, csv_filename) if "csv" in args.log else None

    if log_file is not None or log_stream is not None:
        logger = utils.setup_logger(
            filename=log_file,
            stream=log_stream,
            timestamp=True,
            level=logging.INFO,
        )
        logger.info(args)
    else:
        logger = None
    if is_new_path and "file" in args.log:
        configs.dump_args(osp.join(expr_path, "args.yaml"), args)

    model = load_model(device, args)
    # Since we are training noise, not model
    if args.model != "YOLOv9":
        model = model.eval().requires_grad_(False).to(device)
    else:
        model = model.eval().to(device)

    transform = transforms.Compose(
        [transforms.Resize(args.img_size), transforms.ToTensor()]
    )
    testset = InriaDataset(
        args.data_dir,
        "Test",
        transform=transform,
        in_memory=args.eval,
        transform_cacheable=True,
    )
    # DEBUG: dataset subset
    debug_begin_idx = 0
    debug_num_samples = 16  # len(testset)
    _ = Subset(
        testset, list(range(debug_begin_idx, debug_begin_idx + debug_num_samples))
    )
    testloader = DataLoader(
        testset,
        args.batch_size_test,
        False,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    if args.attack_variant == "lgp":
        adversary = LGP(eps=args.eps)
    elif args.attack_variant == "numbod":
        adversary = NumbOD(eps=args.eps)
    elif args.attack_variant == "default":
        adversary = SimpleNoiseAdversary(eps=args.eps)
    else:
        raise ValueError(args.attack_variant)

    defense_preprocess = get_defense_preprocess(args).eval().to(device)

    # Test last
    avg_map = SamplesMeter(fmt="{:.2f}")
    avg_mar = SamplesMeter(fmt="{:.2f}")
    avg_asr = SamplesMeter(fmt="{:.2f}")
    tqdm_obj = (
        tqdm(total=args.test_repeats * len(testloader))
        if args.defense == "PAD" or log_stream is None
        else None
    )
    for test_idx in range(args.test_repeats):
        if tqdm_obj is not None:
            tqdm_obj.set_description(f"Test repeat {test_idx}")
        meters, map_meter, asr_meter = test_ap(
            model,
            testloader,
            adversary if not args.eval_clean else None,
            device,
            args,
            tqdm_obj,
            preprocess=defense_preprocess,
        )
        if logger is not None:
            logger.info(get_summary(meters))
        map_results: dict[str, Tensor] = map_meter.compute()
        avg_map.update(map_results["map"].item() * 100.0)
        avg_mar.update(map_results["mar_1000"].item() * 100.0)
        avg_asr.update(asr_meter.compute()["asr50"].item() * 100.0)
    if tqdm_obj is not None:
        tqdm_obj.close()

    metric_name = "AP@{:d}".format(int(args.iou_thresh_test * 100))
    msgs = [
        f"{metric_name}: {str(avg_map)}.",
        f"AR@1k: {str(avg_mar)}.",
        f"ASR@50: {str(avg_asr)}.",
    ]
    log_contents(logger, [" ".join(msgs)])  # print in one line


if __name__ == "__main__":
    main()
