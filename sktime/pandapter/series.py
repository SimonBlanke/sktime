"""CompatSeries: pandas Series subclass with cross-version compatibility."""

import pandas as _pd

from sktime.compat.pandas import normalize_freq


class _CompatSeriesMeta(type(_pd.Series)):
    """Metaclass ensuring isinstance works for both compat and plain Series."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, _pd.Series)


class CompatSeries(_pd.Series, metaclass=_CompatSeriesMeta):
    """Version-compatible Series.

    Subclass of ``pd.Series`` that provides automatic frequency alias
    normalization across pandas 1, 2, and 3.
    """

    __module__ = _pd.Series.__module__

    @property
    def _constructor(self):
        return CompatSeries

    @property
    def _constructor_expanddim(self):
        from .frame import CompatDataFrame

        return CompatDataFrame

    def resample(self, rule, *args, **kwargs):
        """Resample with automatic frequency alias normalization."""
        if isinstance(rule, str):
            rule = normalize_freq(rule)
        return super().resample(rule, *args, **kwargs)
