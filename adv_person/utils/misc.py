import hashlib
import os
import os.path as osp
import re
from datetime import datetime
from numbers import Number
from typing import Iterable


def natural_key(string: str, case_sensitive=False) -> list[int | str]:
    """Get human-natural key for sorting.

    Example order: a1.txt, a2.txt, a100.txt
    Implementation is split all number parts and convert to int for sorting.
    Case is not considered. (human rarely use duplicate name with different case,
    and some filesystems even doesn't support it.)
    See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    if not case_sensitive:
        string = string.lower()
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string)]


def sizeof_fmt(num: Number, suffix="B"):
    """Return a human-readable format of large numbers.

    Cr: https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size?page=1&tab=scoredesc#tab-top
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def sizeof_fmt_b1k(num: Number, suffix=""):
    """Return a human-readable format of large numbers.

    Use a base of 1000.
    """
    for unit in ("", "K", "M", "B", "T", "P", "E", "Z"):
        if abs(num) < 1000.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f} Y{suffix}"


def ceil_div(a: int, b: int):
    """d=ceil(a/b).

    Cr: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    """
    return -(a // -b)


def load_class_names(file: str):
    """Load class names from file.

    Load each line as a class.

    Args:
        file (str): filename.

    Returns:
        list[str]: class names.
    """
    with open(file, "r") as f:
        classes = [line.rstrip() for line in f]
    return classes


def filter_not_none(iterable: Iterable):
    """Filter out None values from an iterable.

    Args:
        iterable (Iterable):
            An iterable (e.g., list, tuple) to be filtered.

    Returns:
        filter object:
            An iterator that yields elements from the input iterable excluding
            those that are None.
    """
    return filter(lambda x: x is not None, iterable)


def get_expr_suffix_legacy(root: str, train=True, with_hostname=True):
    idx = 1
    if with_hostname:
        import socket

        prefix = socket.gethostname() + "_"
    else:
        prefix = ""
    while osp.exists(osp.join(root, prefix + str(idx))):
        idx += 1
    return prefix + str(idx if train else idx - 1)


def get_expr_suffix(root: str, name="", train=True, with_hostname=False):
    if train:
        return get_expr_suffix_train(root, name, with_hostname)
    else:
        try:
            all_names = os.listdir(root)
        except FileNotFoundError:
            return ""
        all_names = [s for s in all_names if osp.isdir(osp.join(root, s))]
        if name:
            # filter
            all_names = [s for s in all_names if s.startswith(name)]
        # last match
        all_names.sort(key=natural_key)
        return all_names[-1] if len(all_names) else ""


def get_expr_suffix_train(root: str, name="", with_hostname=False):
    """Generate a unique suffix for a directory or file based on the given root path.

    Args:
        root (str):
            The root directory or file path to which the suffix will be appended.
        name (str, optional):
            Experiment name to be prepended to root.
        train (bool, optional):
            If True, return a new suffix; if False, return the last suffix. Default: True.
        with_hostname (bool, optional):
            If True, the hostname of the machine will be included in the suffix prefix.
            Default: False.

    Returns:
        str:
            A unique suffix that can be appended to the root path.
    """
    idx = 1
    if name:
        fullname = name
    else:
        fullname = datetime.now().strftime("%Y%m%d-%H%M%S")
    if with_hostname:
        import socket

        fullname += "_" + socket.gethostname()
    path = osp.join(root, fullname)
    if osp.exists(path):
        idx = 1
        while osp.exists(osp.join(root, fullname + f"_{idx}")):
            idx += 1
        fullname += f"_{idx}"
    return fullname


def get_file_digest(filename: str, alg="sha256"):
    with open(filename, "rb") as fp:
        digest = hashlib.file_digest(fp, alg)
    return digest
