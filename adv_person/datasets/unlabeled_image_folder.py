import os
import os.path as osp
from typing import Any, Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)


def ls_all_file(path: str, extensions: Optional[list[str]] = None, recursive=False):
    subdirs: list[str] = []
    files: list[str] = []
    for name in os.listdir(path):
        if osp.isdir(osp.join(path, name)):
            subdirs.append(name)
        else:
            if extensions is None or has_file_allowed_extension(name, extensions):
                files.append(name)
    if not recursive:
        return files
    for subdir in subdirs:
        for name in ls_all_file(osp.join(path, subdir), extensions, recursive):
            files.append(osp.join(subdir, name))
    return files


class UnlabeledImageFolder(Dataset):
    """A generic data loader for unlabeled images.

    Args:
        root (string): Root directory path.
        recursive (bool): Whether to go into sub-directories.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of image path
    """

    def __init__(
        self,
        root: str,
        recursive=False,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        super().__init__()
        self.root = osp.expanduser(root)
        self.loader = loader
        self.transform = transform
        self.imgs = ls_all_file(self.root, IMG_EXTENSIONS, recursive)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index) -> Any:
        img = self.loader(osp.join(self.root, self.imgs[index]))
        if self.transform is not None:
            img = self.transform(img)
        return img
