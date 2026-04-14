"""Train camouflage and other 3D-modeling attacks."""

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
from torchvision.datasets import ImageFolder
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import batched_nms, nms
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))
import adv_person.vis.plot as myplt
from adv_person import configs, utils
from adv_person.attacks.modeling import *
from adv_person.configs import CLIConfig
from adv_person.datasets import InriaDataset, detection_collate
from adv_person.defense import EPGFPreprocess
from adv_person.models import *
from adv_person.models.loss import DetectionLoss, NonPrintablityScore, total_variation
from adv_person.utils.image import compute_image_shape
from adv_person.utils.meter import *
from adv_person.utils.patch import init_patch, load_patch
from adv_person.utils.stats_meter import *
from adv_person.utils.typing import _pair, _size_2_t
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
    renderer: RenderState,
    optimizers: list[Optimizer],
    schedulers: list[_LRScheduler],
    device: torch.device,
    args: configs.CLIConfig,
    tqdm_obj: Optional[tqdm] = None,
    nps_calc: Optional[NonPrintablityScore] = None,
    loss_tracker: Optional[BucketHistoryTracker] = None,
    summary_writer: Optional[SummaryWriter] = None,
):
    model.eval()
    criterion = DetectionLoss(
        args.iou_thresh,
        get_obj_cls_loss_fn(args.obj_cls_loss),
        reduction="none",
        loss_type="max_iou",
        pad_zero=True,
    )
    preprocess = get_defense_preprocess(args).to(device)
    target_cls = get_target_cls(args)
    normalize = get_normalization(args).to(device)

    use_tv_loss = args.tv_loss > 0.0
    use_nps_loss = args.nps_loss > 0.0
    begin_iter = len(dataloader) * epoch

    epoch_meter = SimpleMeter("epoch").update(epoch)  # This is to record epoch in log
    lr = SimpleMeter("lr", "{:.2e}")
    train_time = SimpleMeter("train_time", "{:.3f}")
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
            ]
        )
    )

    if loss_tracker is not None:
        renderer.camera_sampler.sampler_probs = (
            loss_tracker.values / loss_tracker.counts
        )
        loss_tracker.decay()

    if tqdm_obj is not None:
        tqdm_obj.set_description(f"Training epoch {epoch}")
    begin_time = time.time()
    for batch_idx, (data, _) in enumerate(dataloader):
        data: Tensor
        data = data.to(device)

        [o.zero_grad(True) for o in optimizers]
        if args.texture == "camouflage":
            tex_kwargs = dict(tau=0.3)
        else:
            tex_kwargs = dict()
        render_kwargs = dict(use_tps2d=args.tps, use_tps3d=args.tps3d)
        data, targets = renderer.forward(
            data,
            resample=batch_idx % 20 == 0 or batch_idx == len(dataloader) - 1,
            share_texture=args.crop == "TCA",
            tex_kwargs=tex_kwargs,
            render_kwargs=render_kwargs,
        )
        if False and args.defense == "EPGF":
            pre_kwargs = dict(soft_factor=4.0 / (epoch + 5))
        else:
            pre_kwargs = dict()
        data_no_preprocess = data
        data, targets = apply_defense_preprocess(
            preprocess, data, targets, **pre_kwargs
        )
        # save_image(data[0].cpu(), "image.png")
        # exit(0)
        all_boxes = get_boxes(model, normalize(data), args)
        # boxes = [box for box, label in all_boxes]
        # Only boxes of target class are considered in loss computation
        boxes = [box[label == target_cls] for box, label in all_boxes]
        det_loss_list = criterion.forward(boxes, targets)
        det_loss = det_loss_list.mean()
        loss = det_loss
        if False and args.defense == "EPGF":
            loss = loss + 0.1 * preprocess.loss(data_no_preprocess)
        if use_tv_loss:
            tv_loss = None
            for texture in renderer.textures:
                if tv_loss is None:
                    tv_loss = texture.loss()
                else:
                    tv_loss += texture.loss()
            loss = loss + args.tv_loss * tv_loss
        if use_nps_loss:
            raise NotImplementedError
            nps_loss = nps_calc.forward(patch)
            loss = loss + args.nps_loss * nps_loss
        loss.backward()
        [o.step() for o in optimizers]
        with torch.no_grad():
            renderer.clamp_(args.clamp_shift)

        if loss_tracker is not None:
            assert len(det_loss_list) == len(renderer.camera_sampler.azim_ind)
            loss_tracker.update(renderer.camera_sampler.azim_ind, det_loss_list)

        train_loss.update(loss.item(), len(data))
        train_loss_det.update(det_loss.item(), len(data))
        if use_tv_loss:
            train_loss_tv.update(tv_loss.item())  # This is not average over inputs
        if use_nps_loss:
            raise NotImplementedError
            train_loss_nps.update(nps_loss.item(), len(data))
        iter = begin_iter + batch_idx + 1
        if summary_writer is not None and iter % args.tensorboard_interval == 0:
            summary_writer.add_scalar("loss/loss", train_loss.val, iter)
            summary_writer.add_scalar("loss/det_loss", train_loss_det.val, iter)
            if use_tv_loss:
                summary_writer.add_scalar("loss/tv_loss", train_loss_tv.val, iter)
            if use_nps_loss:
                summary_writer.add_scalar("loss/nps_loss", train_loss_nps.val, iter)
            lr.update(optimizers[0].param_groups[0]["lr"])
            # lr_seed not logged
            summary_writer.add_scalar("misc/lr", lr.val, iter)
            summary_writer.add_scalar("misc/epoch", epoch, iter)
        if tqdm_obj is not None:
            tqdm_obj.update()
    train_time.update(time.time() - begin_time)
    if args.texture == "camouflage":
        decay_factor = 0.5
        if (epoch + 1) % (args.epochs // 5) == 0:
            # Schedule lr
            for o in optimizers:
                o.param_groups[0]["lr"] *= decay_factor
    else:
        if epoch > 0.4 * args.epochs:
            for s in schedulers:
                s.step(train_loss.result())
    lr.update(optimizers[0].param_groups[0]["lr"])
    # lr_seed not logged
    return meters


@torch.no_grad()
def test_ap(
    model: Module,
    dataloader: DataLoader,
    renderer: RenderState,
    device: torch.device,
    args: configs.CLIConfig,
    tqdm_obj: Optional[tqdm] = None,
    num_samples=36,
):
    model.eval()
    criterion = DetectionLoss(
        args.iou_thresh_test,
        get_obj_cls_loss_fn(args.obj_cls_loss),
        reduction="mean_test",
        loss_type="max_iou",
    )
    preprocess = get_defense_preprocess(args).to(device)
    target_cls = get_target_cls(args)
    normalize = get_normalization(args).to(device)

    asr_meters = [
        utils.AttackSuccessRate(iou_threshold=args.iou_thresh_test)
        for _ in range(num_samples)
    ]
    map_meters = [
        MeanAveragePrecision(iou_thresholds=[args.iou_thresh_test])
        for _ in range(num_samples)
    ]
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

    theta_list = np.linspace(-180, 180, num_samples, endpoint=False)
    # Light is always fixed to AmbientLights
    renderer.lights = renderer.light_sampler.sample(0)

    begin_time = time.time()
    from torchvision.ops import nms
    # from adv_person.datasets.coco_ann import save_coco_cxcywh_ann

    for batch_idx, (data, _) in enumerate(dataloader):
        data: Tensor
        data = data.to(device)
        for angle_idx, theta in enumerate(theta_list):
            # Manually sample cameras
            renderer.cameras = renderer.camera_sampler.sample(len(data), theta=theta)
            if args.texture == "camouflage":
                tex_kwargs = dict(determinate=True)
            else:
                tex_kwargs = dict()
            render_kwargs = dict(use_tps2d=args.tps, use_tps3d=args.tps3d)
            patched, targets = renderer.forward(
                data,
                resample=False,
                is_test=True,
                share_texture=args.crop == "TCA",
                tex_kwargs=tex_kwargs,
                render_kwargs=render_kwargs,
            )

            # DEBUG: print image
            # for i in range(len(patched)):
            #     sample_idx = batch_idx * args.batch_size_test + i
            #     local_target = targets[i]
            #     target_labels = torch.full(
            #         (len(local_target),), target_cls, dtype=torch.int32
            #     )
            #     save_image(
            #         patched[i], f"data/jedi_cleaned/adv_examples/frames/{sample_idx}_{angle_idx}.png"
            #     )
            #     save_coco_cxcywh_ann(
            #         f"data/jedi_cleaned/adv_examples/labels/{sample_idx}_{angle_idx}.txt",
            #         local_target,
            #         target_labels,
            #     )
            # continue

            # TF.to_pil_image(patched[0]).save("image.png")

            if isinstance(preprocess, DWTPreprocessPlugin):
                # Targets not modified in this case
                preprocessed_input, _ = apply_defense_preprocess(
                    preprocess, patched, targets
                )
                level_boxes = []
                boxes = get_boxes(model, normalize(preprocessed_input), args)
                level_boxes.append(boxes)
                extra_levels = args.dwt_extra_levels
                # from adv_person.vis import get_pil_with_bbox
                # get_pil_with_bbox(preprocessed_input[0], boxes[0][0][:,:4], boxes[0][0][:,4], boxes[0][1]).save(f"image{args.dwt_max_level}.png")
                if len(extra_levels) > 0:
                    # Forward multiple levels
                    for level in extra_levels:
                        preprocess.level_override = level
                        preprocessed_input, _ = apply_defense_preprocess(
                            preprocess, patched, targets
                        )
                        boxes = get_boxes(model, normalize(preprocessed_input), args)
                        level_boxes.append(boxes)
                        # get_pil_with_bbox(preprocessed_input[0], boxes[0][0][:,:4], boxes[0][0][:,4], boxes[0][1]).save(f"image{level}.png")
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
                # exit()
            else:
                patched, targets = apply_defense_preprocess(
                    preprocess, patched, targets
                )
                all_boxes = get_boxes(model, normalize(patched), args)
            boxes = [box[idx == target_cls] for box, idx in all_boxes]
            det_loss = criterion.forward(boxes, targets)
            test_loss_det.update(det_loss.item(), len(data))
            for i, (boxes_with_score, cls_idx) in enumerate(all_boxes):
                boxes = boxes_with_score[:, :4]
                scores = boxes_with_score[:, 4]
                # Filter with NMS
                keep = nms(boxes, scores, args.nms_thresh)
                # Filter person class
                keep = keep[cls_idx[keep] == target_cls]
                boxes = boxes[keep]
                scores = scores[keep]
                cls_idx = cls_idx[keep]

                local_target = targets[i]
                target_labels = cls_idx.new_full((len(local_target),), target_cls)
                # The original metric requires absolute box coordinates,
                # but I don't think it would be different for relative ones
                preds = [dict(boxes=boxes, scores=scores, labels=cls_idx)]
                tgts = [dict(boxes=local_target, labels=target_labels)]
                asr_meters[angle_idx].update(preds, tgts)
                map_meters[angle_idx].update(preds, tgts)
                # from adv_person.vis import get_pil_with_bbox
                # get_pil_with_bbox(
                #     patched[i], boxes, scores, cls_idx, colors="magenta"
                # ).save(f"img{i}.png")
            if tqdm_obj is not None:
                tqdm_obj.update()
    test_time.update(time.time() - begin_time)

    anglewise_meter_dict = dict(theta=theta_list, map=map_meters, asr=asr_meters)
    return meters, anglewise_meter_dict


@torch.no_grad()
def load_state_dict(state_dict: dict, textures: list[Module]):
    for i, texture in enumerate(textures):
        strict = True
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        texture._load_from_state_dict(
            state_dict,
            f"texture_{i}.",
            {},
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    f"texture_{i}", "\n\t".join(error_msgs)
                )
            )


def get_state_dict(textures: list[Module], keep_vars=False):
    state_dict = {}
    for i, texture in enumerate(textures):
        texture._save_to_state_dict(state_dict, f"texture_{i}.", keep_vars)
    return state_dict


def make_print_pieces(
    expr_path: str,
    epoch: int,
    person: PersonModel,
    dpi: _size_2_t = 150,
    hs_cm=[75.0, 106.76],
):
    dpi = _pair(dpi)
    px = compute_image_shape((hs_cm[0], None), dpi, person.tshirt.fig_size, is_cm=True)
    TF.to_pil_image(person.tshirt.get_masked_map(px).squeeze(0)).save(
        osp.join(expr_path, f"tshirt{epoch + 1}_print.png"), dpi=dpi
    )
    px = compute_image_shape((hs_cm[1], None), dpi, person.trouser.fig_size, is_cm=True)
    TF.to_pil_image(person.trouser.get_masked_map(px).squeeze(0)).save(
        osp.join(expr_path, f"trouser{epoch + 1}_print.png"), dpi=dpi
    )


@torch.no_grad()
def save_ckpt(
    expr_path: str,
    epoch: int,
    textures: list[ITexture],
):
    camou_tshirt, camou_trouser = textures
    if isinstance(camou_tshirt, CamouflageTexture):
        tex_tshirt = camou_tshirt.forward(determinate=True, transform_color=False)
        tex_trouser = camou_trouser.forward(determinate=True, transform_color=False)
    else:
        tex_tshirt = camou_tshirt.tex_map.detach()
        tex_trouser = camou_trouser.tex_map.detach()
    save_image(
        tex_tshirt.squeeze(0),
        osp.join(expr_path, f"tshirt{epoch + 1}.png"),
    )
    save_image(
        tex_trouser.squeeze(0),
        osp.join(expr_path, f"trouser{epoch + 1}.png"),
    )
    state_dict = get_state_dict(textures)
    torch.save(state_dict, osp.join(expr_path, f"ckpt{epoch + 1}.pt"))
    return tex_tshirt.squeeze(0), tex_trouser.squeeze(0)


def main(args: CLIConfig = None):
    if args is None:
        args = configs.get_args()
    assert args.attack == "render", "Patch-like attacks should use `train_patch.py`"
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
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )
    testset = ImageFolder(
        osp.join(args.data_dir, "background_test"), transform=transform
    )
    if args.defense in ("PAD", "Jedi"):
        testset = Subset(testset, list(range(32, 64)))
    testloader = DataLoader(
        testset,
        args.batch_size_test,
        False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Construct model
    person = PersonModel(args.data_dir, device=device)
    # Color card
    colors: Tensor = torch.load(osp.join(args.data_dir, "camouflage4.pth"))
    colors = colors.float()
    # Pre-computed color transform
    if args.color_transform:
        color_transform = ColorTransform(
            osp.join(args.data_dir, "color_transform_dim6.npz")
        )
    else:
        color_transform = nn.Identity()

    patch_crops: list[Module] = []
    textures: list[ITexture] = []
    for cloth in person.clothes:
        # Compute crop first
        if args.crop == "none":
            map_size = cloth.fig_size
            patch_crop = nn.Identity()
            patch_crop_test = patch_crop
        elif args.crop == "jitter":
            jitter_scale = 1.1
            map_scale = jitter_scale
            # map_scale = 1.0
            crop_size = np.array(cloth.fig_size, dtype=np.int32)
            latent_map_size = (crop_size * jitter_scale).astype(np.int32)
            max_pos = latent_map_size - crop_size
            map_size = (crop_size * map_scale).astype(np.int32)
            pos_shift = (map_size - latent_map_size) // 2
            patch_crop = PatchCropping(
                crop_size.tolist(),
                "random",
                False,
                max_pos=max_pos.tolist(),
                pos_shift=pos_shift.tolist(),
            )
            patch_crop_test = PatchCropping(
                crop_size.tolist(),
                "center",
                False,
                max_pos=max_pos.tolist(),
                pos_shift=pos_shift.tolist(),
            )
            map_size = map_size.tolist()
        elif args.crop == "TCA":
            map_size = (args.patch_size, args.patch_size)
            patch_crop = PatchCropping(cloth.fig_size, "random", True)
            patch_crop_test = patch_crop
        else:
            raise ValueError(args.crop)
        patch_crops.extend([patch_crop, patch_crop_test])

        # Camouflage generator for each piece of clothes
        if args.texture == "camouflage":
            seed_ratio = args.ratio
            num_points = 60
            texture = CamouflageTexture(
                colors, map_size, color_transform, num_points, seed_ratio
            )
            nn.init.uniform_(
                texture.seeds_train, args.clamp_shift, 1 - args.clamp_shift
            )
        elif args.texture == "simple":
            texture = SimpleTexture(map_size, None, color_transform)
            if isinstance(args.patch_init, float):
                nn.init.constant_(texture.tex_map, args.patch_init)
            elif isinstance(args.patch_init, int):
                nn.init.constant_(texture.tex_map, args.patch_init / 255.0)
            elif isinstance(args.patch_init, str):
                if args.patch_init == "random":
                    print("Init random texture")
                    nn.init.uniform_(texture.tex_map)
                else:
                    raise ValueError(args.patch_init)
            else:
                raise ValueError(args.patch_init)
        else:
            raise ValueError(args.texture)
        textures.append(texture.to(device))

    img_synthesizer = ImageSynthesizer(
        contrast=args.contrast,
        brightness=args.brightness,
        noise_factor=args.noise,
        scale=args.scale_range,
        translation=args.translation,
        pooling=args.pooling,
    ).to(device)
    if args.transform_fixed:
        img_synthesizer = img_synthesizer.to_fixed().to(device)
    # img_synthesizer_test = img_synthesizer

    camera_sampler = CameraSampler(device=device)
    light_sampler = LightSampler(device=device)
    renderer = RenderState(
        person,
        camera_sampler,
        light_sampler,
        textures,
        patch_crops,
        img_synthesizer,
    )

    # Load checkpoint
    patch: Optional[dict] = None
    if args.eval_clean:
        # TODO: init the texture to some clean state
        pass
    elif args.patch:
        patch = torch.load(args.patch, "cpu")
        if logger is not None:
            logger.info(f"Load patch file '{args.patch}'")
    elif args.eval:
        if args.eval_best:
            raise NotImplementedError
            ckpt = torch.load(osp.join(expr_path, "patch_best.pt"))
            patch = ckpt["patch"]
            if logger is not None:
                for k, v in ckpt.items():
                    if k != "patch":
                        logger.info(f"{k}: {v}")
                logger.info(f"Load best epoch {ckpt['epoch']}")
        else:
            patch = torch.load(osp.join(expr_path, f"ckpt{args.resume}.pt"), "cpu")
            if logger is not None:
                logger.info(f"Resume epoch {args.resume}")
    if patch is not None:
        load_state_dict(patch, textures)

    if is_train:
        # Save initial checkpoint
        save_ckpt(expr_path, -1, textures)

    if args.nps_loss > 0.0:
        # Reserved
        raise NotImplementedError
        nps_calculator = NonPrintablityScore(args.nps_color_file).to(device)
    else:
        nps_calculator = None

    if is_train:
        trainset = ImageFolder(
            osp.join(args.data_dir, "background"), transform=transform
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
            pin_memory=True,
        )
        camou_tshirt, camou_trouser = textures
        if args.texture == "camouflage":
            optimizer = optim.Adam(
                [camou_tshirt.points, camou_trouser.points], lr=args.lr
            )
            optimizer_seed = optim.Adam(
                [camou_tshirt.seeds_train, camou_trouser.seeds_train], lr=args.lr2
            )
            scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochs // 6, 0.5)
            optimizers = [optimizer, optimizer_seed]
            schedulers = [scheduler]
        else:
            if args.crop == "TCA":
                params = [camou_tshirt.tex_map]
            else:
                params = [camou_tshirt.tex_map, camou_trouser.tex_map]
            optimizer = optim.Adam(params, lr=args.lr, amsgrad=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=20, cooldown=40, min_lr=args.lr / 100
            )
            optimizers = [optimizer]
            schedulers = [scheduler]

        loss_tracker = BucketHistoryTracker(36, device=device)

        # This is to avoid tqdm conflict with log output
        tqdm_obj = (
            tqdm(total=args.epochs * len(trainloader)) if log_stream is None else None
        )
        for epoch in range(args.epochs):
            meters = train(
                epoch,
                model,
                trainloader,
                renderer,
                optimizers,
                schedulers,
                device,
                args,
                tqdm_obj,
                nps_calculator,
                loss_tracker,
                summary_writer,
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

            # We do not save best

            if (epoch + 1) % args.ckpt_interval == 0 or epoch + 1 == args.epochs:
                # Do checkpointing
                tex_tshirt, tex_trouser = save_ckpt(expr_path, epoch, textures)
                if summary_writer is not None:
                    # Also log to tensorboard
                    summary_writer.add_image(
                        "tex_tshirt", tex_tshirt.cpu(), (epoch + 1) * len(trainloader)
                    )
                    summary_writer.add_image(
                        "tex_trouser", tex_trouser.cpu(), (epoch + 1) * len(trainloader)
                    )
        if tqdm_obj is not None:
            tqdm_obj.close()

    # Before testing, make printable pieces
    print_epoch = -1
    if args.resume >= 0:
        print_epoch = args.resume
    elif is_train:
        print_epoch = args.epochs
    if print_epoch >= 0:
        if args.texture == "camouflage":
            tex_kwargs = dict(determinate=True)
        else:
            tex_kwargs = dict()
        tex_kwargs.update(transform_color=False)
        dummy_data = torch.ones((1, 3, 416, 416), device=device)
        renderer.forward(dummy_data, True, True, args.crop == "TCA", tex_kwargs)
        make_print_pieces(
            expr_path,
            print_epoch - 1,
            person,
            dpi=150,
            hs_cm=[75.0 * 1.05, 106.76 * 1.05],
        )
    if False:
        return

    # Test last
    myplt.plotsetup()
    avg_map = SamplesMeter(fmt="{:.2f}")
    avg_asr = SamplesMeter(fmt="{:.2f}")
    tqdm_obj = tqdm(total=args.test_repeats * len(testloader), desc="Testing")
    meters, anglewise_meter_dict = test_ap(
        model,
        testloader,
        renderer,
        device,
        args,
        tqdm_obj,
        args.test_repeats,
    )
    if tqdm_obj is not None:
        tqdm_obj.close()
    if logger is not None:
        logger.info(get_summary(meters))
    maps, asrs = [], []
    thetas = anglewise_meter_dict["theta"]
    for i, theta in enumerate(thetas):
        map_meter: MeanAveragePrecision = anglewise_meter_dict["map"][i]
        asr_meter: utils.AttackSuccessRate = anglewise_meter_dict["asr"][i]
        maps.append(map_meter.compute()["map"].item() * 100.0)
        asrs.append(asr_meter.compute()["asr0"].item() * 100.0)
    avg_map.update(average(maps))
    avg_asr.update(average(asrs))
    # log_contents(logger, [thetas, maps, asrs])
    if osp.exists(expr_path):
        myplt.lineplot(thetas, asrs, "Theta", "ASR (%)")
        myplt.plt.ylim(0, 101)
        myplt.savefig(osp.join(expr_path, "fig_asr_test.png"), dpi=150)
        result_dict = dict(thetas=thetas, maps=np.array(maps), asrs=np.array(asrs))
        np.save(osp.join(expr_path, "test_result.npy"), [result_dict])

    metric_name = "AP@{:d}".format(int(args.iou_thresh_test * 100))
    log_contents(logger, [f"{metric_name}: {str(avg_map)}. ASR@0: {str(avg_asr)}."])


if __name__ == "__main__":
    main()
