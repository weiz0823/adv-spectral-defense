import numpy as np
import scipy.stats


def t_conf_variance(S, n, conf=0.95, tail=2):
    """Variance on student-t distribution.

    Parameters
    ----
    S: Any
        Sample deviation (ddof=1).
    n: int
        Number of samples.
    conf: float
        ``1-alpha``, confidence value.
    tail: 1 | 2
        Single-tailed or two-tailed.

    Returns
    ----
    Same shape as S.
    """
    if tail == 1:
        point = conf
    elif tail == 2:
        point = (1 + conf) / 2
    else:
        raise ValueError("tail should be 1 or 2")
    return scipy.stats.t.ppf(point, n - 1) * S / np.sqrt(n)


def t_conf_interval(a, conf=0.95, tail=2, axis=None):
    """Confidence interval on student-t distribution.

    Parameters
    ----
    a : array_like
        Array containing samples.
    conf: float
        ``1-alpha``, confidence value.
    tail: 1 | 2
        Two-tailed for confidence interval,
        single-tailed for confidence lower or upper bound.
    axis : None or int or tuple of ints, optional
        Axis to compute mean and std.

    Returns
    ----
    x, y: tuple
        If two-tailed, confidence interval is ``x±y``.
        If single-tailed, confidence lower bound is ``x-y``,
        or upper bound is ``x+y``, depending on your hypothesis.
    """
    X_bar = np.mean(a, axis=axis)
    S = np.std(a, axis=axis, ddof=1)
    n = np.size(a, axis=axis)
    return X_bar, t_conf_variance(S, n, conf=conf, tail=tail)


def t_pval(a, mu, test_type=">", axis=None):
    """P-value of single population's mean compared with given mu.

    Parameters
    ----
    a : array_like
        Array containing samples.
    mu: float | array_like
        Target mean value.
    test_type: '>' | '<' | '!='
        Comparison type of alternative hypothesis.
    axis : None or int or tuple of ints, optional
        Axis to compute mean and std.

    Returns
    ----
    P: float | array_like
        P-value.
    """
    valid_test_types = [">", "<", "!="]
    if test_type not in valid_test_types:
        raise ValueError("test_type should be in {}".format(valid_test_types))
    X_bar = np.mean(a, axis=axis)
    S = np.std(a, axis=axis, ddof=1)
    n = np.size(a, axis=axis)
    T = (X_bar - mu) / (S / np.sqrt(n))
    if test_type == "!=":
        T = np.abs(T)
    elif test_type == "<":
        T = -T
    prob = scipy.stats.t.cdf(T)
    if test_type == "!=":
        return 2 * (1 - prob)
    else:
        return 1 - prob


def z_conf_variance(std, n, conf=0.95, tail=2):
    """Variance on normal distribution.

    Parameters
    ----
    std: Any
        Standard deviation of the sample distribution.
    n: int
        Number of samples.
    conf: float
        ``1-alpha``, confidence value.
    tail: 1 | 2
        Single-tailed or two-tailed.

    Returns
    ----
    Same shape as S.
    """
    if tail == 1:
        point = conf
    elif tail == 2:
        point = (1 + conf) / 2
    else:
        raise ValueError("tail should be 1 or 2")
    return scipy.stats.norm.ppf(point) * std / np.sqrt(n)


def z_conf_interval(a, std, conf=0.95, tail=2, axis=None):
    """Confidence interval on normal distribution.

    Parameters
    ----
    a : array_like
        Array containing samples.
    std: Any | None
        Standard deviation of the sample distribution.
        If None, estimate standard deviation using sample deviation.
        Note that student-t distribution should be used if the standard deviation
        is unknown and the number of samples is small.
    conf: float
        ``1-alpha``, confidence value.
    tail: 1 | 2
        Two-tailed for confidence interval,
        single-tailed for confidence lower or upper bound.
    axis : None or int or tuple of ints, optional
        Axis to compute mean and std.

    Returns
    ----
    x, y: tuple
        If two-tailed, confidence interval is ``x±y``.
        If single-tailed, confidence lower bound is ``x-y``,
        or upper bound is ``x+y``, depending on your hypothesis.
    """
    X_bar = np.mean(a, axis=axis)
    if std is None:
        std = np.std(a, axis=axis, ddof=1)
    n = np.size(a, axis=axis)
    return X_bar, z_conf_variance(std, n, conf=conf, tail=tail)
