"""Frequency alias normalization for pandas v1/v2/v3 compatibility.

pandas v2.2 deprecated several DatetimeIndex offset aliases, v3 removes them:

    End-of-period:                Sub-daily:
      M  -> ME  (MonthEnd)          H   -> h   (Hour)
      Q  -> QE  (QuarterEnd)        BH  -> bh  (BusinessHour)
      Y  -> YE  (YearEnd)           CBH -> cbh (CustomBusinessHour)
      BM -> BME (BusinessMonthEnd)  T   -> min (Minute)
      BQ -> BQE (BusinessQtrEnd)    S   -> s   (Second)
      BY -> BYE (BusinessYearEnd)   L   -> ms  (Millisecond)
      SM -> SME (SemiMonthEnd)      U   -> us  (Microsecond)
                                    N   -> ns  (Nanosecond)

    Deprecated old aliases with multiple forms:
      A  -> Y   (was an older name for YearEnd)
      BA -> BY  (was an older name for BusinessYearEnd)
      AS -> YS  (was an older name for YearStart)
      BAS-> BYS (was an older name for BusinessYearStart)

PeriodIndex frequencies ("M", "Q", "Y") are NOT affected by these changes.
Start-of-period aliases (MS, QS, YS, BMS, BQS, BYS) are also unchanged.

sktime uses old-style aliases as its canonical internal representation.
This module converts between old-style and new-style at the pandas boundary.
"""

import re

import pandas as pd

from sktime.utils.pandas_compat._version import pandas_geq

# new-style (v2.2+) -> canonical old-style
_NEW_TO_OLD = {
    "BME": "BM",
    "BQE": "BQ",
    "BYE": "BY",
    "SME": "SM",
    "ME": "M",
    "QE": "Q",
    "YE": "Y",
    "cbh": "CBH",
    "bh": "BH",
    "min": "T",
    "ms": "L",
    "us": "U",
    "ns": "N",
    "h": "H",
    "s": "S",
}

# canonical old-style -> new-style (v2.2+)
_OLD_TO_NEW = {
    "BM": "BME",
    "BQ": "BQE",
    "BY": "BYE",
    "BA": "BYE",
    "SM": "SME",
    "M": "ME",
    "Q": "QE",
    "Y": "YE",
    "A": "YE",
    "CBH": "cbh",
    "BH": "bh",
    "T": "min",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "H": "h",
    "S": "s",
}

# old aliases that had multiple names; normalize to the preferred canonical form
_OLD_SYNONYMS = {
    "A": "Y",
    "BA": "BY",
    "AS": "YS",
    "BAS": "BYS",
}

_FREQ_PATTERN = re.compile(r"^(\d*)([A-Za-z]+)(?:-(.+))?$")


def normalize_freq(freq_str):
    """Normalize a frequency string to sktime's canonical (old-style) form.

    Converts new-style pandas v2.2+ aliases back to old-style, and also
    normalizes deprecated old synonyms (A->Y, BA->BY, AS->YS, BAS->BYS).

    Safe to call on strings that are already in canonical form (idempotent).

    Parameters
    ----------
    freq_str : str or None
        Frequency string, e.g. "ME", "2h", "QE-DEC", "M", "30T".

    Returns
    -------
    str or None
        Canonical old-style frequency string, e.g. "M", "2H", "Q-DEC", "30T".
        Returns None if input is None.

    Examples
    --------
    >>> normalize_freq("ME")
    'M'
    >>> normalize_freq("2h")
    '2H'
    >>> normalize_freq("QE-DEC")
    'Q-DEC'
    >>> normalize_freq("30min")
    '30T'
    >>> normalize_freq("M")   # already canonical, returned unchanged
    'M'
    >>> normalize_freq("A")   # old synonym normalized
    'Y'
    """
    if freq_str is None or not isinstance(freq_str, str):
        return freq_str

    match = _FREQ_PATTERN.match(freq_str)
    if match is None:
        return freq_str

    count, alias, anchor = match.groups()

    mapped = _NEW_TO_OLD.get(alias)
    if mapped is None:
        mapped = _OLD_SYNONYMS.get(alias, alias)

    result = f"{count}{mapped}"
    if anchor:
        result = f"{result}-{anchor}"
    return result


def to_pandas_freq(freq_str):
    """Convert a canonical old-style freq string to the form pandas expects.

    On pandas < 2.2 this returns the string unchanged (old-style is native).
    On pandas >= 2.2 it converts to the new-style aliases that pandas requires.

    Use this when passing frequency strings to DatetimeIndex operations:
    ``pd.date_range``, ``.asfreq()``, ``pd.tseries.frequencies.to_offset()``.

    For PeriodIndex operations (``pd.period_range``, ``.to_period()``,
    ``.to_timestamp()``), pass the canonical old-style string directly
    because Period frequency aliases were not renamed.

    Parameters
    ----------
    freq_str : str or None
        Canonical old-style frequency string, e.g. "M", "2H", "Q-DEC".

    Returns
    -------
    str or None
        Frequency string appropriate for the installed pandas version.

    Examples
    --------
    >>> to_pandas_freq("M")   # on pandas >= 2.2
    'ME'
    >>> to_pandas_freq("2H")  # on pandas >= 2.2
    '2h'
    >>> to_pandas_freq("M")   # on pandas < 2.2
    'M'
    """
    if freq_str is None or not isinstance(freq_str, str):
        return freq_str
    if not pandas_geq(2, 2):
        return freq_str

    match = _FREQ_PATTERN.match(freq_str)
    if match is None:
        return freq_str

    count, alias, anchor = match.groups()
    mapped = _OLD_TO_NEW.get(alias, alias)

    result = f"{count}{mapped}"
    if anchor:
        result = f"{result}-{anchor}"
    return result


def coerce_to_offset(freq_str):
    """Convert a frequency string to a pandas DateOffset object.

    Handles alias conversion for the installed pandas version automatically,
    so neither old-style nor new-style input will produce deprecation warnings.

    Parameters
    ----------
    freq_str : str
        Frequency string in any form (old-style, new-style, or canonical).

    Returns
    -------
    pd.tseries.offsets.BaseOffset
    """
    pandas_freq = to_pandas_freq(normalize_freq(freq_str))
    return pd.tseries.frequencies.to_offset(pandas_freq)


def get_offset_n_and_freqstr(freq_str):
    """Extract multiplier and base frequency from a frequency string.

    Replacement for the pattern ``offset.n, offset.base.freqstr`` which relied
    on ``offset.base`` (deprecated in pandas v2.1, removed in v3).

    The returned base frequency is in canonical old-style form.

    Parameters
    ----------
    freq_str : str
        Frequency string, e.g. "2H", "30T", "M", "QE-DEC".

    Returns
    -------
    n : int
        The multiplier (e.g. 2 for "2H", 1 for "M").
    base_freqstr : str
        Base frequency in canonical old-style form (e.g. "H", "T", "M", "Q-DEC").
    """
    offset = coerce_to_offset(freq_str)
    n = offset.n
    # offset.base.freqstr was the old way; reconstruct by stripping the count prefix
    base_freqstr = re.sub(r"^\d+", "", offset.freqstr)
    return n, normalize_freq(base_freqstr)
