"""Pandapter: pandas adapter providing a version-compatible API for sktime.

Replace ``import pandas as pd`` with::

    from sktime.compat import pandapter as pd

All standard pandas functionality is available transparently. Where the API
differs between pandas 1.x, 2.x, and 3.x (deprecated frequency aliases,
removed methods, renamed functions), pandapter handles the translation
automatically.

Architecture
~~~~~~~~~~~~
pandapter combines three mechanisms:

1. **Module proxy** (``__getattr__``): anything not explicitly overridden is
   forwarded to the real pandas module. ``pd.concat``, ``pd.to_datetime``,
   ``pd.errors``, etc. all work without any extra code.

2. **Wrapped constructors**: ``pd.date_range``, ``pd.period_range`` and
   similar functions normalize frequency alias strings before passing them
   to pandas.

3. **Compat subclasses**: ``CompatDataFrame``, ``CompatSeries``, and
   ``CompatDatetimeIndex`` add version-safe method overrides (e.g.
   ``applymap``, ``resample`` with freq normalization, ``freq`` setter).
   These are constructed by pandapter and propagated through pandas
   operations via ``_constructor``.

At sktime's base-class entry points (``_check_X_y``), incoming plain
``pd.DataFrame`` objects are converted to ``CompatDataFrame`` via
:func:`ensure_compat`. At exit points (``predict``, ``transform``),
:func:`ensure_native` converts back to plain pandas so users never see
the compat types.
"""

import pandas as _pd

from sktime.compat.pandas import normalize_freq

from . import api  # noqa: F401 -- makes pd.api.types resolve to our subpackage
from .frame import CompatDataFrame
from .indexes import CompatDatetimeIndex
from .series import CompatSeries

__all__ = [
    "DataFrame",
    "Series",
    "CompatDataFrame",
    "CompatSeries",
    "CompatDatetimeIndex",
    "date_range",
    "period_range",
    "concat",
    "merge",
    "ensure_compat",
    "ensure_native",
]

# Expose compat subclasses as the default constructors.
# The metaclass on each ensures isinstance checks work bidirectionally:
#   isinstance(plain_pd_df, pandapter.DataFrame)  ->  True
#   isinstance(compat_df, pd.DataFrame)           ->  True
DataFrame = CompatDataFrame
Series = CompatSeries


def date_range(*args, **kwargs):
    """``pd.date_range`` with automatic frequency alias normalization.

    Returns a :class:`CompatDatetimeIndex` whose ``freq`` setter also
    normalizes aliases.
    """
    if "freq" in kwargs and isinstance(kwargs["freq"], str):
        kwargs["freq"] = normalize_freq(kwargs["freq"])
    result = _pd.date_range(*args, **kwargs)
    if type(result) is _pd.DatetimeIndex:
        result.__class__ = CompatDatetimeIndex
    return result


def period_range(*args, **kwargs):
    """``pd.period_range`` passed through without offset alias normalization.

    PeriodIndex uses its own frequency naming that is separate from offset
    aliases. ``"M"`` means monthly period and stays ``"M"`` across all pandas
    versions (pandas explicitly rejects ``"ME"`` for periods).
    """
    return _pd.period_range(*args, **kwargs)


def concat(objs, *args, **kwargs):
    """``pd.concat`` that returns a compat-typed result."""
    result = _pd.concat(objs, *args, **kwargs)
    return ensure_compat(result)


def merge(*args, **kwargs):
    """``pd.merge`` that returns a compat-typed result."""
    result = _pd.merge(*args, **kwargs)
    return ensure_compat(result)


def ensure_compat(obj):
    """Convert a plain pandas object to its compat version.

    Creates a shallow copy (no data duplication) and reassigns its
    ``__class__``. The original object is never mutated, which is
    important for sktime's side-effect-free contract on fit/transform args.

    Parameters
    ----------
    obj : pd.DataFrame, pd.Series, or None
        The object to convert.

    Returns
    -------
    CompatDataFrame, CompatSeries, or None
        A shallow copy with its class upgraded, or the original if
        already a compat type or not a pandas object.
    """
    if obj is None:
        return None
    if type(obj) is _pd.DataFrame:
        obj = obj.copy(deep=False)
        obj.__class__ = CompatDataFrame
        _ensure_compat_index(obj)
    elif type(obj) is _pd.Series:
        obj = obj.copy(deep=False)
        obj.__class__ = CompatSeries
        _ensure_compat_index(obj)
    return obj


def ensure_native(obj):
    """Convert a compat object back to a plain pandas type.

    The inverse of :func:`ensure_compat`.  Use at exit points where
    results are returned to the user.

    Parameters
    ----------
    obj : CompatDataFrame, CompatSeries, or None

    Returns
    -------
    pd.DataFrame, pd.Series, or None
    """
    if obj is None:
        return None
    if type(obj) is CompatDataFrame:
        _ensure_native_index(obj)
        obj.__class__ = _pd.DataFrame
    elif type(obj) is CompatSeries:
        _ensure_native_index(obj)
        obj.__class__ = _pd.Series
    return obj


def _ensure_compat_index(obj):
    """Upgrade DatetimeIndex instances on *obj* to CompatDatetimeIndex.

    Creates a shallow copy of the index before reassigning its class so
    that the original object's index (which may be shared via shallow copy)
    is never mutated.
    """
    if not hasattr(obj, "index"):
        return
    idx = obj.index
    if isinstance(idx, _pd.MultiIndex):
        needs_update = False
        new_levels = list(idx.levels)
        for i, level in enumerate(new_levels):
            if type(level) is _pd.DatetimeIndex:
                new_level = level.copy()
                new_level.__class__ = CompatDatetimeIndex
                new_levels[i] = new_level
                needs_update = True
        if needs_update:
            obj.index = idx.set_levels(new_levels)
    elif type(idx) is _pd.DatetimeIndex:
        new_idx = idx.copy()
        new_idx.__class__ = CompatDatetimeIndex
        obj.index = new_idx


def _ensure_native_index(obj):
    """Revert CompatDatetimeIndex instances on *obj* to plain DatetimeIndex."""
    if not hasattr(obj, "index"):
        return
    idx = obj.index
    if isinstance(idx, _pd.MultiIndex):
        for level in idx.levels:
            if type(level) is CompatDatetimeIndex:
                level.__class__ = _pd.DatetimeIndex
    elif type(idx) is CompatDatetimeIndex:
        idx.__class__ = _pd.DatetimeIndex


def __getattr__(name):
    """Forward any attribute not defined above to the real pandas module."""
    return getattr(_pd, name)
