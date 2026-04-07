"""Compat index classes with version-compatible freq handling."""

import pandas as _pd

from sktime.compat.pandas import normalize_freq

# Locate the freq property descriptor in the DatetimeIndex class hierarchy.
# We need the raw descriptor (not its value) so we can call fget/fset
# from our overriding property.
_dt_freq_prop = None
for _cls in _pd.DatetimeIndex.__mro__:
    if "freq" in getattr(_cls, "__dict__", {}):
        _candidate = _cls.__dict__["freq"]
        if hasattr(_candidate, "fget"):
            _dt_freq_prop = _candidate
            break


class CompatDatetimeIndex(_pd.DatetimeIndex):
    """DatetimeIndex with a freq setter that normalizes deprecated aliases.

    Setting ``index.freq = "M"`` on pandas >= 2.1 is transparently
    converted to ``"ME"``, and similarly for other renamed aliases.
    The getter delegates to the original pandas property unchanged.
    """

    if _dt_freq_prop is not None:

        @property
        def freq(self):
            """Frequency of the index, delegating to the pandas implementation."""
            return _dt_freq_prop.fget(self)

        if _dt_freq_prop.fset is not None:

            @freq.setter
            def freq(self, value):
                if isinstance(value, str):
                    value = normalize_freq(value)
                _dt_freq_prop.fset(self, value)
