"""Pandas version compatibility layer for sktime.

Provides a stable internal API for pandas functionality that differs across
pandas 1.x, 2.x, and 3.x. All version-specific pandas logic is concentrated
here so the rest of sktime can remain version-agnostic.

Import from ``sktime.compat.pandas`` instead of using version-dependent
pandas APIs directly.

Version support: pandas >= 1.1, < 4.0
"""

import re
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

__all__ = [
    "PANDAS_VERSION",
    "PANDAS_GE_110",
    "PANDAS_GE_150",
    "PANDAS_GE_200",
    "PANDAS_GE_210",
    "PANDAS_GE_220",
    "PANDAS_GE_300",
    "is_float_dtype",
    "is_integer_dtype",
    "is_signed_integer_dtype",
    "is_unsigned_integer_dtype",
    "is_bool_dtype",
    "is_string_dtype",
    "is_numeric_dtype",
    "is_datetime64_any_dtype",
    "is_timedelta64_dtype",
    "is_categorical_dtype",
    "offset_freq",
    "period_freq",
    "normalize_freq",
    "df_map",
    "set_freq",
    "ensure_index",
    "is_nested_object",
    "suppress_freq_warning",
]


# ---------------------------------------------------------------------------
# Version detection (computed once at import time)
# ---------------------------------------------------------------------------

_version_match = re.match(r"(\d+)\.(\d+)\.(\d+)", pd.__version__)
if _version_match is None:
    raise RuntimeError(f"Cannot parse pandas version: {pd.__version__!r}")

PANDAS_VERSION = tuple(int(x) for x in _version_match.groups())

PANDAS_GE_110 = PANDAS_VERSION >= (1, 1, 0)
PANDAS_GE_150 = PANDAS_VERSION >= (1, 5, 0)
PANDAS_GE_200 = PANDAS_VERSION >= (2, 0, 0)
PANDAS_GE_210 = PANDAS_VERSION >= (2, 1, 0)
PANDAS_GE_220 = PANDAS_VERSION >= (2, 2, 0)
PANDAS_GE_300 = PANDAS_VERSION >= (3, 0, 0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_dtype(arr_or_dtype):
    """Extract dtype from an array-like, Series, Index, or dtype object.

    Handles numpy dtype instances, numpy scalar type classes (e.g. ``np.float64``),
    pandas ExtensionDtype objects, and any object with a ``.dtype`` attribute.
    """
    if isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype
    # numpy scalar type classes like np.float64, np.int32 etc.
    if isinstance(arr_or_dtype, type) and issubclass(arr_or_dtype, np.generic):
        return np.dtype(arr_or_dtype)
    # Python built-in numeric types (float, int, complex, bool)
    if isinstance(arr_or_dtype, type):
        try:
            return np.dtype(arr_or_dtype)
        except TypeError:
            pass
    if hasattr(arr_or_dtype, "dtype"):
        return arr_or_dtype.dtype
    return arr_or_dtype


# ---------------------------------------------------------------------------
# Dtype checks
#
# Drop-in replacements for the pd.api.types.is_*_dtype family.
# These functions were deprecated in pandas 2.1 and are scheduled for removal.
# Our implementations use numpy dtype.kind and pandas ExtensionDtype.kind,
# both of which are stable across all supported versions.
# ---------------------------------------------------------------------------


def is_float_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a float dtype.

    Replacement for ``pd.api.types.is_float_dtype``.
    Recognises both numpy float dtypes and pandas nullable Float types.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind == "f"
    return getattr(dtype, "kind", None) == "f"


def is_integer_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of an integer dtype.

    Includes both signed and unsigned integers.
    Replacement for ``pd.api.types.is_integer_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind in ("i", "u")
    return getattr(dtype, "kind", None) in ("i", "u")


def is_signed_integer_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a signed integer dtype.

    Replacement for ``pd.api.types.is_signed_integer_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind == "i"
    return getattr(dtype, "kind", None) == "i"


def is_unsigned_integer_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of an unsigned integer dtype.

    Replacement for ``pd.api.types.is_unsigned_integer_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind == "u"
    return getattr(dtype, "kind", None) == "u"


def is_bool_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a boolean dtype.

    Replacement for ``pd.api.types.is_bool_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind == "b"
    return getattr(dtype, "kind", None) == "b"


def is_string_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a string dtype.

    Returns True for numpy object/unicode/bytes dtypes and for pandas
    StringDtype. Returns False for CategoricalDtype even though its
    internal kind is ``"O"``.

    Works across pandas 1-3: on pandas 3 where strings default to
    ``pd.StringDtype`` instead of ``object``, this still returns True.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if isinstance(dtype, np.dtype):
        return dtype.kind in ("U", "S", "O")
    if hasattr(pd, "StringDtype") and isinstance(dtype, pd.StringDtype):
        return True
    kind = getattr(dtype, "kind", None)
    return kind in ("U", "S", "O")


def is_numeric_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a numeric dtype.

    Numeric includes integer, unsigned integer, float, and complex.
    Boolean is NOT considered numeric (matching pandas convention).

    Replacement for ``pd.api.types.is_numeric_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind in ("i", "u", "f", "c")
    return getattr(dtype, "kind", None) in ("i", "u", "f", "c")


def is_datetime64_any_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a datetime64 dtype.

    Includes both timezone-naive (``datetime64[ns]``) and timezone-aware
    (``DatetimeTZDtype``) variants, as well as all resolutions
    (``ns``, ``us``, ``ms``, ``s``).

    Replacement for ``pd.api.types.is_datetime64_any_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind == "M"
    return getattr(dtype, "kind", None) == "M"


def is_timedelta64_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a timedelta64 dtype.

    Replacement for ``pd.api.types.is_timedelta64_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    if isinstance(dtype, np.dtype):
        return dtype.kind == "m"
    return getattr(dtype, "kind", None) == "m"


def is_categorical_dtype(arr_or_dtype):
    """Check whether the provided array or dtype is of a categorical dtype.

    Replacement for ``pd.api.types.is_categorical_dtype``.
    """
    dtype = _get_dtype(arr_or_dtype)
    return isinstance(dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------------------
# Frequency alias mappings
#
# Pandas renamed many offset frequency aliases in two waves:
#   2.1: month/quarter/year-end offsets (M -> ME, Q -> QE, Y -> YE, ...)
#   2.2: time-unit aliases (H -> h, T -> min, S -> s, L -> ms, ...)
# The old aliases were removed in pandas 3.0.
# ---------------------------------------------------------------------------

# Wave 1 (pandas 2.1): month/quarter/year-end offsets
_FREQ_ALIASES_21 = {
    "M": "ME",
    "BM": "BME",
    "SM": "SME",
    "CBM": "CBME",
    "Q": "QE",
    "BQ": "BQE",
    "Y": "YE",
    "BY": "BYE",
    "A": "YE",
    "BA": "BYE",
    "AS": "YS",
    "BAS": "BYS",
}

# Wave 2 (pandas 2.2): time-unit aliases
_FREQ_ALIASES_22 = {
    "H": "h",
    "BH": "bh",
    "CBH": "cbh",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
}

# Reverse mappings (new -> old) for backward conversion on older pandas.
# Built with setdefault so that "A" -> "YE" doesn't override "Y" -> "YE".
_FREQ_ALIASES_21_REV = {}
for _old, _new in _FREQ_ALIASES_21.items():
    _FREQ_ALIASES_21_REV.setdefault(_new, _old)

_FREQ_ALIASES_22_REV = {v: k for k, v in _FREQ_ALIASES_22.items()}

# regex for decomposing compound freq strings like "2ME", "Q-FEB", "3min"
_FREQ_PATTERN = re.compile(r"^(\d*)([A-Za-z]+)(-[A-Z]+)?$")


def normalize_freq(freq_str):
    """Normalize a frequency alias to the format expected by the installed pandas.

    Converts between old-style (``M``, ``H``, ``T``, ...) and new-style
    (``ME``, ``h``, ``min``, ...) frequency aliases. Handles compound
    strings like ``"2M"``, ``"3H"``, or anchored strings like ``"Q-FEB"``.

    Pass-through for strings that are not recognized as aliases, for
    ``None``, and for non-string inputs (e.g. ``DateOffset`` objects).

    Parameters
    ----------
    freq_str : str, None, or other
        The frequency alias to normalize. ``None`` and non-strings are
        returned unchanged.

    Returns
    -------
    str or None
        The normalized frequency alias appropriate for the installed pandas.
    """
    if freq_str is None or not isinstance(freq_str, str):
        return freq_str

    match = _FREQ_PATTERN.match(freq_str)
    if not match:
        return freq_str

    prefix, base, anchor = match.groups()
    anchor = anchor or ""

    # Forward conversion: old -> new (on pandas >= 2.1 / >= 2.2)
    if PANDAS_GE_210 and base in _FREQ_ALIASES_21:
        base = _FREQ_ALIASES_21[base]
    if PANDAS_GE_220 and base in _FREQ_ALIASES_22:
        base = _FREQ_ALIASES_22[base]

    # Backward conversion: new -> old (on pandas < 2.1 / < 2.2)
    if not PANDAS_GE_210 and base in _FREQ_ALIASES_21_REV:
        base = _FREQ_ALIASES_21_REV[base]
    if not PANDAS_GE_220 and base in _FREQ_ALIASES_22_REV:
        base = _FREQ_ALIASES_22_REV[base]

    return prefix + base + anchor


# Canonical name -> old-style base alias for offset frequencies.
_OFFSET_CANONICAL = {
    "month_end": "M",
    "business_month_end": "BM",
    "semi_month_end": "SM",
    "custom_business_month_end": "CBM",
    "quarter_end": "Q",
    "business_quarter_end": "BQ",
    "year_end": "Y",
    "business_year_end": "BY",
    "year_start": "AS",
    "business_year_start": "BAS",
    "hour": "H",
    "business_hour": "BH",
    "custom_business_hour": "CBH",
    "minute": "T",
    "second": "S",
    "millisecond": "L",
    "microsecond": "U",
    "nanosecond": "N",
    "day": "D",
    "business_day": "B",
    "week": "W",
    "month_start": "MS",
    "business_month_start": "BMS",
    "quarter_start": "QS",
    "business_quarter_start": "BQS",
}


def offset_freq(name):
    """Return the version-appropriate offset frequency alias.

    Use this for ``pd.date_range``, ``pd.DataFrame.resample``,
    ``pd.tseries.frequencies.to_offset``, and similar APIs that accept
    DateOffset frequency strings.

    Parameters
    ----------
    name : str
        Canonical frequency name. Valid names:
        ``month_end``, ``quarter_end``, ``year_end``, ``hour``,
        ``minute``, ``second``, ``day``, ``week``, etc.
        See ``_OFFSET_CANONICAL`` for the full list.

    Returns
    -------
    str
        The frequency alias string appropriate for the installed pandas.

    Raises
    ------
    ValueError
        If *name* is not a recognized canonical frequency name.

    Examples
    --------
    >>> from sktime.compat.pandas import offset_freq
    >>> idx = pd.date_range("2020", periods=3, freq=offset_freq("month_end"))
    """
    if name not in _OFFSET_CANONICAL:
        raise ValueError(
            f"Unknown offset frequency name {name!r}. "
            f"Valid names: {sorted(_OFFSET_CANONICAL.keys())}"
        )
    return normalize_freq(_OFFSET_CANONICAL[name])


# Canonical name -> old-style base alias for period frequencies.
# Month, quarter, and year are period-native concepts and use different
# aliases than the offset equivalents (e.g. "M" for monthly period stays "M",
# it does not become "ME" like the MonthEnd offset).
_PERIOD_CANONICAL = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "week": "W",
    "day": "D",
    "hour": "H",
    "minute": "T",
    "second": "S",
    "millisecond": "L",
    "microsecond": "U",
    "nanosecond": "N",
}


def period_freq(name):
    """Return the version-appropriate period frequency alias.

    Use this for ``pd.PeriodIndex``, ``pd.period_range``, and
    ``Series.dt.to_period``.

    For month, quarter, and year the alias does not change across pandas
    versions (these are period-native frequencies, not offset aliases).
    For sub-daily units the same renaming as offsets applies
    (``H`` -> ``h``, ``T`` -> ``min``, etc.).

    Parameters
    ----------
    name : str
        Canonical period name: ``year``, ``quarter``, ``month``, ``week``,
        ``day``, ``hour``, ``minute``, ``second``, ``millisecond``,
        ``microsecond``, ``nanosecond``.

    Returns
    -------
    str
        The period frequency alias string.

    Raises
    ------
    ValueError
        If *name* is not a recognized period frequency name.

    Examples
    --------
    >>> from sktime.compat.pandas import period_freq
    >>> idx = pd.period_range("2020-01", periods=3, freq=period_freq("month"))
    """
    if name not in _PERIOD_CANONICAL:
        raise ValueError(
            f"Unknown period frequency name {name!r}. "
            f"Valid names: {sorted(_PERIOD_CANONICAL.keys())}"
        )
    old_alias = _PERIOD_CANONICAL[name]
    # Period-native frequencies (M, Q, Y) are not affected by the offset
    # alias rename, only the time-unit aliases (H, T, S, ...) are.
    if old_alias in _FREQ_ALIASES_22:
        return normalize_freq(old_alias)
    return old_alias


# ---------------------------------------------------------------------------
# DataFrame operation utilities
# ---------------------------------------------------------------------------


def df_map(df):
    """Return the element-wise map method for a DataFrame.

    ``DataFrame.applymap`` was deprecated in pandas 2.1 and removed in 3.0
    in favour of ``DataFrame.map``. This function returns whichever method
    is available on the installed pandas version.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose map method to return.

    Returns
    -------
    callable
        ``df.map`` if available, otherwise ``df.applymap``.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2]})
    >>> result = df_map(df)(str)
    """
    if hasattr(df, "map"):
        return df.map
    return df.applymap


def set_freq(index, freq, level=None):
    """Set frequency on an Index or a specific MultiIndex level.

    Returns the modified index. Callers should reassign::

        df.index = set_freq(df.index, offset_freq("day"))

    For MultiIndex levels, ``set_levels`` is used to avoid mutating shared
    level arrays (which breaks under Copy-on-Write).

    No automatic normalization is applied because the correct alias depends
    on context (offset vs period). Use :func:`offset_freq`, :func:`period_freq`,
    or :func:`normalize_freq` before calling this function.

    Parameters
    ----------
    index : pd.Index or pd.MultiIndex
        The index to modify.
    freq : str, DateOffset, or None
        The frequency to set. Use ``offset_freq``/``period_freq`` to obtain
        the version-appropriate string.
    level : int or None, default None
        If not None and *index* is a MultiIndex, set the frequency on this
        level only.

    Returns
    -------
    pd.Index or pd.MultiIndex
        The index with updated frequency.
    """
    if level is not None and isinstance(index, pd.MultiIndex):
        new_level = index.levels[level].copy()
        new_level.freq = freq
        return index.set_levels(new_level, level=level)

    index.freq = freq
    return index


def ensure_index(values):
    """Convert *values* to a ``pd.Index`` if it is not one already.

    Replacement for the private ``pandas.core.indexes.base.ensure_index``.
    """
    if isinstance(values, pd.Index):
        return values
    return pd.Index(values)


def is_nested_object(obj):
    """Check whether a Series contains nested pandas Series as values.

    Returns ``True`` if *obj* is an object-dtype Series and at least one of
    its values is itself a ``pd.Series``.

    Replacement for the private ``pandas.core.dtypes.cast.is_nested_object``.
    """
    if not isinstance(obj, pd.Series):
        return False
    if obj.dtype != np.dtype("O"):
        return False
    if len(obj) == 0:
        return False
    return any(isinstance(v, pd.Series) for v in obj.values)


# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------


@contextmanager
def suppress_freq_warning():
    """Context manager that suppresses FutureWarning about deprecated freq aliases.

    On pandas 2.1-2.x the old offset aliases (``M``, ``H``, ``T``, ...)
    trigger ``FutureWarning``. On pandas < 2.1 no warnings are emitted,
    and on pandas >= 3.0 the old aliases are removed entirely (errors, not
    warnings). This context manager is therefore only active on 2.1 <= pandas < 3.0.
    """
    if PANDAS_GE_210 and not PANDAS_GE_300:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*is deprecated and will be removed in a future version",
                category=FutureWarning,
            )
            yield
    else:
        yield
