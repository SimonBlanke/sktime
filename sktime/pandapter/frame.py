"""CompatDataFrame: pandas DataFrame subclass with cross-version compatibility.

Overrides only the methods that differ across pandas 1.x, 2.x, and 3.x.
All other DataFrame behavior is inherited unchanged from pandas.
"""

import pandas as _pd

from sktime.compat.pandas import normalize_freq


class _CompatDataFrameMeta(type(_pd.DataFrame)):
    """Metaclass ensuring isinstance works for both compat and plain DataFrames.

    With this metaclass, ``isinstance(any_pd_dataframe, CompatDataFrame)``
    returns True. This allows pandapter to expose ``DataFrame = CompatDataFrame``
    while keeping isinstance checks working for plain pandas DataFrames
    that haven't been converted yet.
    """

    def __instancecheck__(cls, instance):
        return isinstance(instance, _pd.DataFrame)


class CompatDataFrame(_pd.DataFrame, metaclass=_CompatDataFrameMeta):
    """Version-compatible DataFrame.

    Subclass of ``pd.DataFrame`` that provides stable method names and
    automatic frequency alias normalization across pandas 1, 2, and 3.
    """

    # sktime's mtype system checks type(obj).__module__ to verify objects
    # originate from the pandas package. Without this, CompatDataFrame objects
    # are rejected because their module would be "sktime.pandapter.frame".
    __module__ = _pd.DataFrame.__module__

    @property
    def _constructor(self):
        return CompatDataFrame

    @property
    def _constructor_sliced(self):
        from .series import CompatSeries

        return CompatSeries

    def applymap(self, func, **kwargs):
        """Element-wise function application across all cells.

        ``DataFrame.applymap`` was renamed to ``DataFrame.map`` in pandas 2.1
        and removed in pandas 3.0. This method delegates to whichever
        implementation exists on the installed pandas version.
        """
        if hasattr(_pd.DataFrame, "map"):
            return self.map(func, **kwargs)
        return _pd.DataFrame.applymap(self, func, **kwargs)

    def resample(self, rule, *args, **kwargs):
        """Resample with automatic frequency alias normalization.

        Converts deprecated aliases like ``"M"`` to ``"ME"`` on pandas >= 2.1.
        """
        if isinstance(rule, str):
            rule = normalize_freq(rule)
        return super().resample(rule, *args, **kwargs)
