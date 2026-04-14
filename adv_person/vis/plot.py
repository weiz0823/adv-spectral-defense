from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig

SNS_THEME_DEFAULT = {
    "style": "ticks",
    "rc": {},  # matplotlib.rcParams
}


def plotsetup(cfg=SNS_THEME_DEFAULT):
    sns.set_theme(**cfg)


def _get_fig(fig) -> Figure:
    return fig if fig is not None else plt.gcf()


def _get_ax(ax) -> plt.Axes:
    return ax if ax is not None else plt.gca()


def clear(ax=None):
    _get_ax(ax).clear()


def multiplot(x, y, ax=None, **kwargs):
    ax = _get_ax(ax)
    if not isinstance(x, np.ndarray) and isinstance(x[0], Iterable):
        for xi, yi in zip(x, y):
            ax.plot(xi, yi, **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    return ax


def lineplot(
    x,
    y,
    xlabel=None,
    ylabel=None,
    legend=None,
    ax=None,
    all_args: dict = {},
):
    """Simple utility to gather all functions to plot single or multiple lines."""
    ax = _get_ax(ax)
    multiplot(x, y, ax=ax, **all_args.get("plot", {}))
    if xlabel is not None:
        ax.set_xlabel(xlabel, **all_args.get("xlabel", {}))
    if ylabel is not None:
        ax.set_ylabel(ylabel, **all_args.get("ylabel", {}))
    if legend is not None:
        ax.legend(legend, **all_args.get("legend", {}))
    return ax
