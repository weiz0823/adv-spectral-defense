import fnmatch
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset

from .coco_ann import *
from .paddings import *
from .pascal_ann import *


class InriaDataset(VisionDataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.

    Args:
        pad: square | none
    """

    def __init__(
        self,
        root: str,
        split="Test",
        lab_dir: str | None = None,
        max_lab=16,
        pad="square",
        transforms=None,
        transform=None,
        target_transform=None,
        in_memory=False,
        transform_cacheable=False,
    ):
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        assert pad in ("square", "none")
        self.pad = pad
        self.split = split
        self.split_dir = os.path.join(self.root, self.split)
        # Positive samples are used
        self.img_dir = os.path.join(self.split_dir, "pos")
        self.img_names = fnmatch.filter(
            os.listdir(self.img_dir), "*.png"
        ) + fnmatch.filter(os.listdir(self.img_dir), "*.jpg")
        self.img_names.sort()  # Sort for reproducibility
        n_images = len(self.img_names)
        if lab_dir is not None:
            # Use custom labels, assuming MS-COCO annotation format
            self.lab_dir = os.path.join(lab_dir, self.split)
            self.lab_fmt = "coco"
            n_labels = len(fnmatch.filter(os.listdir(self.lab_dir), "*.txt"))
            assert n_images == n_labels, (
                "Number of images and number of labels don't match"
            )
        else:
            # Use dataset labels
            self.lab_dir = os.path.join(self.split_dir, "annotations")
            self.lab_fmt = "pascal"
        self.img_paths: list[str] = []
        self.lab_paths: list[str] = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
            lab_name = img_name.rsplit(".", 1)[0] + ".txt"
            self.lab_paths.append(os.path.join(self.lab_dir, lab_name))
        self.max_n_labels = max_lab

        # Preload into memory
        self.in_memory = in_memory
        self.transform_cacheable = transform_cacheable
        if self.in_memory:
            if self.transform_cacheable:
                self.raw_items = [
                    self.transforms(*self._get_raw_item(i))
                    for i in range(self.__len__())
                ]
            else:
                self.raw_items = [self._get_raw_item(i) for i in range(self.__len__())]
        else:
            self.raw_items = None

    def __len__(self):
        return len(self.img_names)

    def _get_raw_item(self, idx: int):
        assert idx <= len(self), "index range error"
        img_path = self.img_paths[idx]
        lab_path = self.lab_paths[idx]
        image: Image.Image = Image.open(img_path).convert("RGB")

        # Class labels ignored because all ground truths are persons
        if self.lab_fmt == "coco":
            bboxes, _ = load_coco_ann(lab_path)
        else:
            bboxes, _ = load_pascal_ann(lab_path)

        label = torch.tensor(bboxes)
        # TODO: we can split padding transform out
        if self.pad == "square":
            if self.lab_fmt == "coco":
                # Generated labels are already padded
                # FIXME: we should avoid padded labels for compatibility
                image = pad_to_square(image)
            else:
                image, label = pad_to_square(image, label)
        # label = self.pad_lab(label)
        return image, label

    def __getitem__(self, idx: int):
        if self.in_memory:
            if self.transform_cacheable:
                return self.raw_items[idx]
            else:
                image, label = self.raw_items[idx]
        else:
            image, label = self._get_raw_item(idx)
        return self.transforms(image, label)
