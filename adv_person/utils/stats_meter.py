from .meter import Meter
from .stats import *


class DistributionMeter(Meter):
    """Track all values to compute distribution parameters."""

    def __init__(self, name="", fmt="{}"):
        super().__init__(name, fmt)
        self.val = None
        self.mean = None
        self.std = None
        self.all_val = []

    def update(self, val, n=1):
        self.val = val
        self.all_val.append(val)
        return self

    def collect(self):
        self.mean = np.mean(self.all_val)
        if len(self.all_val) > 1:
            self.std = np.std(self.all_val, ddof=1)
        else:
            self.std = 0
        return self

    def result(self):
        self.collect()
        return self.mean

    def __str__(self):
        self.collect()
        fmtstr_name = "{name}=" if self.name else ""
        fmtstr = (
            fmtstr_name
            + self.fmt.replace("{", "{mean")
            + "±"
            + self.fmt.replace("{", "{std")
        )
        return fmtstr.format(**self.__dict__)


class SamplesMeter(Meter):
    """Track all values to compute confidence interval."""

    def __init__(self, name="", fmt="{}", conf=0.95):
        super().__init__(name, fmt)
        self.conf = conf
        self.val = None
        self.mean = None
        self.conf_std = None
        self.all_val = []

    def update(self, val, n=1):
        self.val = val
        self.all_val.append(val)
        return self

    def collect(self):
        if len(self.all_val) > 1:
            self.mean, self.conf_std = t_conf_interval(self.all_val, conf=self.conf)
        else:
            self.mean = np.mean(self.all_val)
            self.conf_std = 0

    def result(self):
        self.collect()
        return self.mean

    def __str__(self):
        self.collect()
        fmtstr_name = "{name}=" if self.name else ""
        if len(self.all_val) > 1:
            fmtstr = (
                fmtstr_name
                + self.fmt.replace("{", "{mean")
                + "±"
                + self.fmt.replace("{", "{conf_std")
            )
        else:
            fmtstr = fmtstr_name + self.fmt.replace("{", "{mean")
        return fmtstr.format(**self.__dict__)
