import os
import os.path as osp

from adv_person import configs, utils


def get_tensorboard_name(args: configs.CLIConfig, expr_path: str):
    if args.defense == "none":
        defense_name = "nodefense"
    else:
        defense_name = args.defense
    suffix = osp.split(expr_path)[1]
    return "_".join([defense_name, suffix])


def _get_expr_path_patch(args: configs.CLIConfig):
    # Add attack
    attack_name = args.attack
    texture_name = args.texture
    # Add crop
    if args.crop == "none":
        crop_name = "no_crop"
    else:
        crop_name = args.crop
    # Add model
    model_name = args.model
    # Add defense
    if args.defense == "none":
        defense_name = "no_defense"
    else:
        defense_name = args.defense
    # Add loss
    loss_name = args.obj_cls_loss
    return [attack_name, texture_name, crop_name, model_name, defense_name, loss_name]


def _get_expr_path_camou(args: configs.CLIConfig):
    # Add attack
    attack_name = args.attack
    texture_name = args.texture
    # Add crop
    if args.crop == "none":
        crop_name = "no_crop"
    else:
        crop_name = args.crop
    # Add model
    model_name = args.model
    # Add defense
    if args.defense == "none":
        defense_name = "no_defense"
    else:
        defense_name = args.defense
    # Add seed ratio
    sr_name = f"sr{args.ratio * 100:.0f}"
    return [attack_name, texture_name, crop_name, model_name, defense_name, sr_name]


def get_expr_path_legacy(args: configs.CLIConfig, train=True, alternative=True):
    if args.attack == "render":
        all_names = _get_expr_path_camou(args)
    else:
        all_names = _get_expr_path_patch(args)
    expr_path = osp.join(args.output_dir, *all_names)
    suffix = args.expr_suffix or utils.get_expr_suffix_legacy(expr_path, train)
    if (
        not train
        and alternative
        and not args.expr_suffix
        and osp.exists(expr_path)
        and not osp.exists(osp.join(expr_path, suffix))
    ):
        # Path exists but not suffix, search for alternative suffix
        # Expected case is training on a different host and test use that index
        # If index matters, this may not be expected. Use user-defined index then (never changed automatically).
        suffix = sorted(os.listdir(expr_path))[-1]
    return osp.join(expr_path, suffix)


def get_expr_path(args: configs.CLIConfig, train=True, alternative=True):
    if args.attack in ("render", "patch"):
        return get_expr_path_legacy(args, train=train, alternative=alternative)

    expr_path = args.output_dir
    suffix = utils.get_expr_suffix(expr_path, args.expr_suffix, train)
    return osp.join(expr_path, suffix)
