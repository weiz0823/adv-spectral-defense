"""Train patch-like attack, and also RCA texture or else, on INRIA Person dataset."""

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
from adv_person.attacks.patch import *
from adv_person.configs import CLIConfig
from adv_person.datasets import InriaDataset, detection_collate
from adv_person.models import *
from adv_person.models.loss import DetectionLoss, NonPrintablityScore, total_variation
from adv_person.utils.meter import *
from adv_person.utils.patch import init_patch, load_patch
from adv_person.utils.stats_meter import *
from adv_person.utils.torch_utils import gen_uniform_grid
from tools.common import *
from tools.expr_path import *

try:
    from adv_person.models.mmdet import *
except ImportError:
    logging.warning("MMDetection models unavailable!")


def train(
    epoch: int,
    model: Module,
    dataloader: DataLoader,
    patch: Tensor,
    patch_applier: PatchProcessorAndApplier,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    args: configs.CLIConfig,
    tqdm_obj: Optional[tqdm] = None,
    nps_calc: Optional[NonPrintablityScore] = None,
    summary_writer: Optional[SummaryWriter] = None,
    preprocess: Optional[Module] = None,
):
    _WARNED_PATCH_NAN_OR_INF = False
    model.eval()
    criterion = DetectionLoss(
        args.iou_thresh, get_obj_cls_loss_fn(args.obj_cls_loss), reduction="mean"
    )
    if preprocess is None:
        preprocess = get_defense_preprocess(args).to(device)
    if args.defense == "NAPGuard":
        preprocess.train()
    normalize = get_normalization(args).to(device)

    use_tv_loss = args.tv_loss > 0.0
    use_nps_loss = args.nps_loss > 0.0
    begin_iter = len(dataloader) * epoch

    epoch_meter = SimpleMeter("epoch").update(epoch)  # This is to record epoch in log
    lr = SimpleMeter("lr", "{:.2e}")
    train_time = SimpleMeter("train_time", "{:.3f}")
    train_saturation = None  # SimpleMeter("saturation", "{:.3f}")
    train_loss = AverageMeter("train_loss", "{:.3f}")
    train_loss_det = AverageMeter("train_loss_det", "{:.3f}")
    train_loss_tv = AverageMeter("train_loss_tv", "{:.3f}") if use_tv_loss else None
    train_loss_nps = AverageMeter("train_loss_nps", "{:.3f}") if use_nps_loss else None
    meters: list[Meter] = list(
        utils.filter_not_none(
            [
                epoch_meter,
                train_time,
                lr,
                train_loss,
                train_loss_det,
                train_loss_tv,
                train_loss_nps,
                train_saturation,
            ]
        )
    )

    if tqdm_obj is not None:
        tqdm_obj.set_description(f"Training epoch {epoch}")
    begin_time = time.time()
    n_repeats = 1
    for batch_idx, (data, target, target_lens) in enumerate(dataloader):
        data: Tensor
        target: Tensor
        target_lens: Tensor
        if n_repeats > 1:
            data = data.repeat(n_repeats, 1, 1, 1)
            target = target.repeat(n_repeats, 1)
            target_lens = target_lens.repeat(n_repeats)
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad(True)
        data = patch_applier.forward(patch, data, target, target_lens)
        data_before_defense = data
        data, target = apply_defense_preprocess(preprocess, data, target)
        all_boxes: list[tuple[Tensor, Tensor]] = get_boxes(model, normalize(data), args)
        boxes = [box for box, label in all_boxes]
        det_loss = criterion.forward(boxes, target.split(list(target_lens)))
        loss = det_loss
        if use_tv_loss:
            tv_loss = torch.clamp_min(args.tv_loss * total_variation(patch), 0.1)
            loss = loss + tv_loss
        if use_nps_loss:
            nps_loss = nps_calc.forward(patch)
            loss = loss + args.nps_loss * nps_loss
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            if not _WARNED_PATCH_NAN_OR_INF and not torch.all(torch.isfinite(patch)):
                _WARNED_PATCH_NAN_OR_INF = True
                logging.warning("NaN or Inf found in optimized patch.")
            patch.nan_to_num_(0, 0, 0)
            patch.clamp_(0, 1)  # Keep patch in image range

        train_loss.update(loss.item(), len(data))
        train_loss_det.update(det_loss.item(), len(data))
        if use_tv_loss:
            train_loss_tv.update(tv_loss.item(), len(data))
        if use_nps_loss:
            train_loss_nps.update(nps_loss.item(), len(data))
        iter = begin_iter + batch_idx + 1
        if summary_writer is not None and iter % args.tensorboard_interval == 0:
            summary_writer.add_scalar("loss/loss", train_loss.val, iter)
            summary_writer.add_scalar("loss/det_loss", train_loss_det.val, iter)
            if use_tv_loss:
                summary_writer.add_scalar("loss/tv_loss", train_loss_tv.val, iter)
            if use_nps_loss:
                summary_writer.add_scalar("loss/nps_loss", train_loss_nps.val, iter)
            lr.update(optimizer.param_groups[0]["lr"])
            summary_writer.add_scalar("misc/lr", lr.val, iter)
            summary_writer.add_scalar("misc/epoch", epoch, iter)
        if tqdm_obj is not None:
            tqdm_obj.update()
    train_time.update(time.time() - begin_time)
    if train_saturation is not None:
        saturation = utils.rgb_to_hsv(patch.detach().cpu()).unbind(0)[1]
        train_saturation.update(saturation.mean().item())
    if epoch > 0.4 * args.epochs:
        scheduler.step(train_loss.result())
    lr.update(optimizer.param_groups[0]["lr"])
    return meters


@torch.no_grad()
def test_ap(
    model: Module,
    dataloader: DataLoader,
    patch: Optional[Tensor],
    patch_applier: PatchProcessorAndApplier,
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
        if patch is not None:
            data = patch_applier.forward(patch, data, target, target_lens)
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

            # DEBUG: print image
            # for i in range(len(data)):
            #     sample_idx = batch_idx * args.batch_size_test + i
            #     name = dataloader.dataset.img_names[sample_idx]
            #     save_image(data[i], f"frames/{name}")
            # exit()

            all_boxes: list[tuple[Tensor, Tensor]] = get_boxes(
                model, normalize(data), args
            )
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
            # from adv_person.vis import get_pil_with_bbox
            # get_pil_with_bbox(data[i], boxes, scores, cls_idx, colors="magenta").save(
            #     f"img{i}.png"
            # )
        if tqdm_obj is not None:
            tqdm_obj.update()
    test_time.update(time.time() - begin_time)

    return meters, map_meter, asr_meter


def main(args: CLIConfig = None):
    if args is None:
        args = configs.get_args()
    assert args.attack == "patch", "3D rendering attacks should use `train_camou.py`"
    if args.eval_best or args.eval_clean:
        args.eval = True
    assert args.eval or args.resume < 0, "Resume on training to be implemented"
    assert (
        not args.eval
        or args.resume >= 0
        or args.eval_best
        or args.patch
        or args.eval_clean
    ), "There must be some patch to load for testing"

    # Here are some coupled arguments set for convenience
    if args.eval and "tensorboard" in args.log:
        args.log = ["stream"]
    if args.eval_clean:
        args.test_repeats = 1

    is_train = not args.eval
    is_eval = args.eval
    device = utils.get_device()
    print("Device is", device, file=sys.stderr)
    expr_path = get_expr_path(args, is_train)
    print(f"Experiment path is '{expr_path}'", file=sys.stderr)
    if is_train:
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
    if is_train:
        configs.dump_args(osp.join(expr_path, "args.yaml"), args)

    model = load_model(device, args)
    # Since we are training patch, not model
    if args.model == "YOLOv9":
        model = model.eval().to(device)
    else:
        model = model.eval().requires_grad_(False).to(device)

    transform = transforms.Compose(
        [transforms.Resize(args.img_size), transforms.ToTensor()]
    )
    testset = InriaDataset(
        args.data_dir,
        "Test",
        # lab_dir=args.label_dir,
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

    # Load or init patch
    if args.eval_clean:
        # We construct the patch here, but we will not pass it into test function
        patch = torch.zeros((3, args.patch_size, args.patch_size), device=device)
    elif args.patch:
        patch = load_patch(args.patch, args.patch_size)
        if logger is not None:
            logger.info(f"Load patch file '{args.patch}'")
        if patch.ndim == 4:
            patch.squeeze_(0)
        if (
            args.patch.endswith("patch_init.bmp")
            and isinstance(args.patch_init, str)
            and args.patch_init == "random"
        ):
            print("Init random patch")
            patch.uniform_()
        patch = patch.to(device)
    elif args.eval:
        if args.eval_best:
            ckpt = torch.load(osp.join(expr_path, "patch_best.pt"))
            patch = ckpt["patch"]
            if logger is not None:
                for k, v in ckpt.items():
                    if k != "patch":
                        logger.info(f"{k}: {v}")
                logger.info(f"Load best epoch {ckpt['epoch']}")
        else:
            patch = TF.to_tensor(
                Image.open(osp.join(expr_path, f"patch{args.resume}.bmp"))
            )
            if logger is not None:
                logger.info(f"Resume epoch {args.resume}")
        patch = patch.to(device)
    else:
        # Init patch for training
        patch = torch.empty((3, args.patch_size, args.patch_size), device=device)
        # patch = torch.empty((3, 300, 200), device=device)
        init_patch(patch, args.patch_init)
    patch.requires_grad_()

    if is_train:
        # Save initial patch
        patch_ckpt = patch.detach().requires_grad_(False)
        ckpt_file = osp.join(expr_path, "patch_init.bmp")
        save_image(patch_ckpt, ckpt_file)

    patch_transformer = PatchTransformer(
        args.img_size,
        contrast=args.contrast,
        brightness=args.brightness,
        angle=args.angle,
        noise_factor=args.noise,
        rand_loc=args.lc_scale > 0,
        lc_scale=args.lc_scale,
        pooling=args.pooling,
        scale=args.scale,
        y_ratio=args.ratio,
        old_fashion=args.old_fashion,
    ).to(device)
    if args.transform_fixed:
        patch_transformer = patch_transformer.to_fixed_transform().to(device)
    patch_transformer_test = patch_transformer  # .to_fixed_transform()

    if args.crop == "none":
        # Base attack without cropping
        patch_cropping = None
    elif args.crop == "RCA":
        patch_cropping = PatchCropping(args.crop_size, "random", False)
    elif args.crop == "TCA":
        patch_cropping = PatchCropping(args.crop_size, "random", True)
    else:
        raise ValueError(args.attack)
    patch_cropping_test = patch_cropping

    if args.tps:
        target_ctrl_points = gen_uniform_grid((-1, -1), (1, 1), (5, 5), flatten=True)
        patch_size = args.crop_size if args.crop_size is not None else args.patch_size
        tps_sampler = ThinPlateSpline(patch_size, target_ctrl_points).to(device)
    else:
        tps_sampler = None
    if args.nps_loss > 0.0:
        nps_calculator = NonPrintablityScore(args.nps_color_file).to(device)
    else:
        nps_calculator = None

    patch_applier = PatchProcessorAndApplier(
        patch_transformer, patch_cropping, tps_sampler
    )
    # Following adversarial texture's setting, TPS is not used in testing
    patch_applier_test = PatchProcessorAndApplier(
        patch_transformer_test, patch_cropping_test
    )

    defense_preprocess = get_defense_preprocess(args).eval().to(device)

    if is_train:
        trainset = InriaDataset(
            args.data_dir,
            "Train",
            # lab_dir=args.label_dir,
            transform=transform,
            in_memory=True,
            transform_cacheable=True,
        )
        # Dataset subset
        if args.subset > 0:
            idxs = list(range(len(trainset)))
            random.shuffle(idxs)
            trainset = Subset(trainset, idxs[: args.subset])
        trainloader = DataLoader(
            trainset,
            args.batch_size,
            True,
            num_workers=args.workers,
            collate_fn=detection_collate,
            pin_memory=True,
        )
        optimizer = optim.Adam([patch], lr=args.lr, amsgrad=True)
        # optimizer = optim.AdamW([patch], lr=args.lr, amsgrad=True, weight_decay=5e-3)
        # optimizer = optim.SGD([patch], lr=args.lr, momentum=0.8, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=30, cooldown=200, min_lr=args.lr / 100
        )

        best_train_loss = torch.inf

        # This is to avoid tqdm conflict with log output
        tqdm_obj = (
            tqdm(total=args.epochs * len(trainloader)) if log_stream is None else None
        )
        for epoch in range(args.epochs):
            meters = train(
                epoch,
                model,
                trainloader,
                patch,
                patch_applier,
                optimizer,
                scheduler,
                device,
                args,
                tqdm_obj,
                nps_calculator,
                summary_writer,
                preprocess=defense_preprocess,
            )
            results = get_results(meters)

            if logger is not None:
                logger.info(get_summary(meters))
            if csv_file is not None:
                # Log csv
                with open(csv_file, "w") as f:
                    if epoch == 0:
                        # csv header
                        print(get_summary(meters, "{name}", ","), file=f)
                    print(get_summary(meters, "csv"), file=f)

            patch_ckpt = patch.detach().requires_grad_(False)
            # DEBUG: we do not save best
            if epoch > args.epochs // 2 and results["train_loss"] < best_train_loss:
                # Save best
                best_train_loss = results["train_loss"]
                ckpt = {"epoch": epoch, "patch": patch_ckpt}
                for k, v in results.items():
                    if "loss" in k:
                        ckpt[k] = v
                ckpt_file = osp.join(expr_path, "patch_best.pt")
                torch.save(ckpt, ckpt_file)

            if (epoch + 1) % args.ckpt_interval == 0 or epoch + 1 == args.epochs:
                # Do checkpointing
                # BMP is readable and not lossfully compressed
                ckpt_file = osp.join(expr_path, f"patch{epoch + 1}.bmp")
                save_image(patch_ckpt, ckpt_file)
                if summary_writer is not None:
                    # Also log to tensorboard
                    summary_writer.add_image(
                        "patch", patch_ckpt.cpu(), (epoch + 1) * len(trainloader)
                    )
        if tqdm_obj is not None:
            tqdm_obj.close()

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
            patch if not args.eval_clean else None,
            patch_applier_test,
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
