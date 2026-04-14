import os.path as osp


def user_path(path: str):
    """User path type."""
    return osp.expanduser(path)


def int_or_float(num: str):
    try:
        return int(num)
    except ValueError:
        return float(num)
