import fnmatch
import os
import os.path as osp
from typing import Optional

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
from .pascal_ann import *


def find_all_images(root: str, recurse=True):
    imgs: list[str] = []
    for p in os.listdir(root):
        fullp = osp.join(root, p)
        if recurse and osp.isdir(fullp):
            sub_images = find_all_images(fullp, recurse)
            for sub_img in sub_images:
                imgs.append(osp.join(p, sub_img))
        elif fullp.endswith((".png", ".jpg")):
            imgs.append(p)
    return imgs


class PhysicalDataset(VisionDataset):
    """A physical dataset.

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.

    Args:
        subdirs: list of sub-directories, if you want a subset of the dataset.
        pad: square | none
    """

    def __init__(
        self,
        root: str,
        subdirs: Optional[list[str]] = None,
        max_lab=1,
        pad="none",
        transform=None,
        target_transform=None,
        in_memory=False,
        transform_cacheable=False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert pad in ("square", "square_crop", "none")
        self.pad = pad
        # Positive samples are used
        self.img_dir = osp.join(self.root, "frames")
        self.lab_dir = osp.join(self.root, "labels")
        if subdirs is None or len(subdirs) == 0:
            self.img_names = find_all_images(self.img_dir, True)
        else:
            self.img_names = []
            for sub in subdirs:
                subnames = find_all_images(osp.join(self.img_dir, sub), True)
                for name in subnames:
                    self.img_names.append(osp.join(sub, name))
        n_images = len(self.img_names)
        self.lab_fmt = "coco-c"
        self.in_memory = in_memory
        self.transform_cacheable = transform_cacheable
        self.img_paths: list[str] = []
        self.lab_paths: list[str] = []
        for img_name in self.img_names:
            self.img_paths.append(osp.join(self.img_dir, img_name))
            lab_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
            self.lab_paths.append(osp.join(self.lab_dir, lab_name))
        self.max_n_labels = max_lab

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
        if self.lab_fmt == "coco-c":
            bboxes, _ = load_coco_cxcywh_ann(lab_path)
        elif self.lab_fmt == "coco":
            bboxes, _ = load_coco_ann(lab_path)
        else:
            bboxes, _ = load_pascal_ann(lab_path)

        label = torch.tensor(bboxes)
        if self.pad == "square":
            if self.lab_fmt == "coco":
                # Generated labels are already padded
                image, _ = self.pad_to_square(image, None)
            else:
                image, label = self.pad_to_square(image, label)
        elif self.pad == "square_crop":
            image, label = self.crop_to_square(image, label)
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

    def crop_to_square(self, img: Image.Image, bboxes: Tensor | None):
        """Crop an image to square and move its bounding boxes correspondingly.

        Args:
            img: Input image
            bboxes: Input bounding boxes

        Returns:
            tuple of padded img and moved bboxes
        """
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            if w < h:
                padding = (h - w) // 2
                padded_img = img.crop((0, padding, w, w + padding))
                if bboxes is not None:
                    # Move ymin, ymax
                    bboxes[:, (1, 3)] = (bboxes[:, (1, 3)] * h - padding) / w
                    bboxes.clamp_(0, 1)
            else:
                padding = (w - h) // 2
                padded_img = img.crop((padding, 0, h + padding, h))
                if bboxes is not None:
                    # Move xmin, xmax
                    bboxes[:, (0, 2)] = (bboxes[:, (0, 2)] * w - padding) / h
                    bboxes.clamp_(0, 1)
        return padded_img, bboxes

    def pad_to_square(self, img: Image.Image, bboxes: Tensor | None):
        """Pad an image to square and move its bounding boxes correspondingly.

        Args:
            img: Input image
            bboxes: Input bounding boxes

        Returns:
            tuple of padded img and moved bboxes
        """
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            if w < h:
                padding = (h - w) // 2
                padded_img = Image.new("RGB", (h, h), color=(127, 127, 127))
                padded_img.paste(img, (padding, 0))
                if bboxes is not None:
                    # Move xmin, xmax
                    bboxes[:, (0, 2)] = (bboxes[:, (0, 2)] * w + padding) / h
            else:
                padding = (w - h) // 2
                padded_img = Image.new("RGB", (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, padding))
                if bboxes is not None:
                    # Move ymin, ymax
                    bboxes[:, (1, 3)] = (bboxes[:, (1, 3)] * h + padding) / w
        return padded_img, bboxes

    def pad_lab(self, lab: Tensor):
        """Pad labels to maximum length."""
        pad_size = self.max_n_labels - lab.shape[0]
        padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=-1) if pad_size > 0 else lab
        return padded_lab
