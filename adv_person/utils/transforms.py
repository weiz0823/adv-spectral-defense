import torchvision.transforms.functional as TF
from torch.nn import Module


def square_crop(img):
    _, h, w = TF.get_dimensions(img)
    size = min(h, w)
    return TF.center_crop(img, (size, size))


def center_resized_crop(img, size: tuple[int, int], scale=1.0, **kwargs):
    _, h, w = TF.get_dimensions(img)
    H, W = size
    if h * W > H * w:
        # Original h/w bigger
        target_size = (int(H * w / W + 0.5), w)
    else:
        target_size = (h, int(W * h / H + 0.5))
    if scale < 1.0:
        hh, ww = target_size
        target_size = (int(scale * hh), int(scale * ww))
    # First get the target H/W, then resize to target size
    return TF.resize(TF.center_crop(img, target_size), size, **kwargs)


class CenterResizedCrop(Module):
    def __init__(self, size: tuple[int, int], scale=1.0, **kwargs):
        super().__init__()
        self.size = size
        self.scale = scale
        self.kwargs = kwargs

    def forward(self, img):
        return center_resized_crop(img, self.size, self.scale, **self.kwargs)
