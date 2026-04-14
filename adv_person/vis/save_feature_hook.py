from torch import Tensor
from torch.nn import BatchNorm2d, Module


class SaveFeatureHook(object):
    def __init__(self, m: Module):
        self.m = m
        self.remove_handle = self.m.register_forward_hook(self.save_feature_hook_fn)

    def save_feature_hook_fn(
        self, module: Module, input: tuple[Tensor], output: Tensor
    ):
        self.in_feature = input[0].detach()
        self.out_feature = output.detach()

    def remove_hook(self):
        self.remove_handle.remove()


class SaveBNFeatureHook(object):
    def __init__(self, m: BatchNorm2d):
        self.m = m
        self.remove_handle = self.m.register_forward_hook(self.save_feature_hook_fn)

    def save_feature_hook_fn(
        self, module: BatchNorm2d, input: tuple[Tensor], output: Tensor
    ):
        module = module.bn
        self.in_feature = input[0].detach()
        # Expects N(0, 1)
        self.out_feature = (
            output.detach() - module.bias.reshape(-1, 1, 1)
        ) / module.weight.reshape(-1, 1, 1)

    def remove_hook(self):
        self.remove_handle.remove()
