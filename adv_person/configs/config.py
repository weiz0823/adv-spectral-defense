import os.path as osp
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import yaml

from .cli_config import *


class BaseCLIConfig(Namespace):
    config: list[str]


def get_base_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        "--cfg",
        "-c",
        action="extend",
        metavar="FILE",
        nargs="+",
        default=[],
        help="YAML config file(s) specifying default arguments",
    )
    return parser


def load_yaml_args(fn: str):
    """Load args in yaml file recursively.

    A 'base_config' field is reserved, which support either a string for filename,
    or a list of filenames, which are the parent configs that are automatically loaded.
    However, using a list of configs is recommended, since it better supports combination
    of configs of different parts.

    Args:
        fn (str): filename.
        parser (ArgumentParser): parser to store arguments.

    Returns:
        args (dict): the args loaded.
    """
    path = Path(fn)
    yaml_args: dict = yaml.safe_load(path.read_text())

    # Base config is still supported, but in some cases, file list is more convenient
    base_files: list[str] | str = yaml_args.pop("base_config", [])
    if isinstance(base_files, str):
        base_files = [base_files]
    all_args = {}
    for base_file in base_files:
        # Recursive load
        all_args.update(load_yaml_args(path.parent / base_file))

    all_args.update(yaml_args)
    return all_args


def get_args(args: Optional[list[str]] = None) -> CLIConfig:
    """Get arguments.

    Overriding hierarchy is (priority low to high):
      - Namespace default
      - Argument parser default
      - Yaml files (latter overrides former)
      - Command line argument
    """
    base_parser = get_base_parser()
    base_args: BaseCLIConfig
    base_args, remaining = base_parser.parse_known_args(args, namespace=BaseCLIConfig())
    parser = get_parser()
    for cfg in base_args.config:
        parser.set_defaults(**load_yaml_args(cfg))
    return parser.parse_args(remaining, namespace=CLIConfig())


def dump_args(fn: str, args: CLIConfig):
    """Dump all arguments into yaml file."""
    with open(fn, "w") as f:
        yaml.safe_dump(args.__dict__, f)
