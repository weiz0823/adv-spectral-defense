# The torchvision nms operator seems to be C++ implemented.
# It's a good choice to use that.
from torchvision.ops import batched_nms, nms
