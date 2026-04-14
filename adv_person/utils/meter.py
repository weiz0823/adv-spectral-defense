from enum import Enum
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class Meter(object):
    def __init__(self, name="", fmt="{}"):
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def summary(self) -> str:
        return self.__str__()


class SimpleMeter(Meter):
    """Always stores the current value"""

    def __init__(self, name="", fmt="{}"):
        super().__init__(name, fmt)
        self.val = None

    def update(self, val, n=1):
        self.val = val
        return self

    def result(self):
        return self.val

    def __str__(self):
        fmtstr_name = "{name}=" if self.name else ""
        fmtstr = fmtstr_name + self.fmt.replace("{", "{val")
        return fmtstr.format(**self.__dict__)


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt="{}", summary_type=Summary.AVERAGE):
        super().__init__(name, fmt)
        self.summary_type = summary_type
        self.val = None
        self.avg = None
        self.sum = None
        self.count: int = 0

    def update(self, val, n=1):
        self.val = val
        if self.sum is None:
            self.sum = val * n
        else:
            self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self

    def result(self):
        if self.summary_type is Summary.NONE:
            val = self.val
        elif self.summary_type is Summary.AVERAGE:
            val = self.avg
        elif self.summary_type is Summary.SUM:
            val = self.sum
        elif self.summary_type is Summary.COUNT:
            val = self.count
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return val

    def __str__(self):
        fmtstr_name = "{name}=" if self.name else ""
        fmtstr = (
            fmtstr_name
            + self.fmt.replace("{", "{val")
            + " ("
            + self.fmt.replace("{", "{avg")
            + ")"
        )
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name}=" + self.fmt.replace("{", "{avg")
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name}=" + self.fmt.replace("{", "{sum")
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name}=" + self.fmt.replace("{", "{count")
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class TensorAverageMeter(AverageMeter):
    """This class could help to eliminate gpu synchronization across batches."""

    def __init__(
        self,
        name="",
        fmt="{}",
        summary_type=Summary.AVERAGE,
        device=torch.device("cpu"),
        dtype: torch.dtype = None,
    ):
        super().__init__(name, fmt, summary_type)
        self._sum = torch.zeros((1,), dtype=dtype, device=device)

    def update(self, val, n=1):
        self._sum += val * n
        self.count += n

    def collect(self):
        self.sum = self._sum.item()
        self.avg = self.sum / self.count

    def result(self):
        self.collect()
        return super().result()

    def __str__(self):
        self.collect()
        return super().__str__()

    def summary(self):
        self.collect()
        return super().summary()


class TensorAccuracyMeter(TensorAverageMeter):
    """This class could help to eliminate gpu synchronization across batches."""

    def __init__(
        self,
        name="",
        fmt="{}",
        summary_type=Summary.AVERAGE,
        device=torch.device("cpu"),
        dtype: torch.dtype = torch.int,
    ):
        super().__init__(name, fmt, summary_type, device, dtype)

    def update(self, val, n=1):
        self._sum += val
        self.count += n

    def collect(self):
        super().collect()
        self.avg = self.avg * 100.0


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: Iterable[Meter], prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class BucketHistoryTracker(object):
    _EPS = 1e-5

    def __init__(self, num_buckets: int, decay_factor=0.5, device=None) -> None:
        self.values = torch.ones(num_buckets, device=device)
        self.counts = torch.ones(num_buckets, device=device)
        self.decay_factor = decay_factor
        self.device = device

    def update(self, indices: Tensor, values: Tensor):
        self.values.index_put_((indices,), values, accumulate=True)
        self.counts.index_put_((indices,), torch.ones_like(values), accumulate=True)
        return self

    def decay(self):
        self.values = self.values * self.decay_factor + self._EPS
        self.counts = self.counts * self.decay_factor + self._EPS
        return self


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk: tuple[int] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred: Tensor
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_results(meters: Iterable[Meter]):
    return dict((m.name, m.result()) for m in meters)


def get_summary(meters: Iterable[Meter], format="{name}={val}", delim="  "):
    r"""Get summary string from meters.

    Suggested formats:
      - format="{name}={val}", delim="  "
      - format="{val}", delim="\t" (note: use format="{name}", delim="\t" to get header)
      - format="csv" (note: delim is ignored and set to "," in csv format)
    """
    if format == "csv":
        return ",".join(m.result() for m in meters)
    l = []
    for m in meters:
        l.append(
            format.replace("{name}", m.name).replace("{val}", m.fmt.format(m.result()))
        )
    return delim.join(l)
