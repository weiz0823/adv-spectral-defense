# https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

plt.rcParams["savefig.bbox"] = "tight"


def show_images(imgs: list[Tensor] | Tensor):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach().cpu()
        img = F.to_pil_image(img)
        axs[0][i].imshow(np.asarray(img))
        axs[0][i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
