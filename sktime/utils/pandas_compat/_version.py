"""Pandas version detection, evaluated once at import time."""

import pandas as pd

PANDAS_VERSION = (int(pd.__version__.split(".")[0]), int(pd.__version__.split(".")[1]))


def pandas_geq(major, minor=0):
    """Check whether the installed pandas version is >= major.minor.

    Parameters
    ----------
    major : int
    minor : int, default 0

    Returns
    -------
    bool
    """
    return PANDAS_VERSION >= (major, minor)
