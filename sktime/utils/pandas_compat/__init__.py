"""Pandas v1/v2/v3 compatibility layer for sktime.

sktime uses old-style frequency aliases as its canonical internal form.
Conversion to/from the pandas-version-specific form happens at the boundary,
i.e., when reading from or passing to pandas APIs.

Typical usage in sktime code:

    from sktime.utils.pandas_compat import normalize_freq, to_pandas_freq

    # reading a freq string from pandas (normalize to canonical old-style)
    freq = normalize_freq(index.freqstr)

    # passing a freq string to a pandas API (convert to version-appropriate form)
    rng = pd.date_range("2020", periods=10, freq=to_pandas_freq("M"))
"""

from sktime.utils.pandas_compat._dtype import (
    coerce_nullable_dtypes,
    is_nullable_numeric_dtype,
    make_numeric_dataframe,
)
from sktime.utils.pandas_compat._freq import (
    coerce_to_offset,
    get_offset_n_and_freqstr,
    normalize_freq,
    to_pandas_freq,
)
from sktime.utils.pandas_compat._version import (
    PANDAS_VERSION,
    pandas_geq,
)

__all__ = [
    "PANDAS_VERSION",
    "coerce_nullable_dtypes",
    "coerce_to_offset",
    "get_offset_n_and_freqstr",
    "is_nullable_numeric_dtype",
    "make_numeric_dataframe",
    "normalize_freq",
    "pandas_geq",
    "to_pandas_freq",
]
