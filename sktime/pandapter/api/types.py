"""Version-compatible pandas.api.types replacements.

Provides our own implementations for dtype-checking functions that were
deprecated in pandas 2.1 and removed in pandas 3.0. Everything else
delegates to the real pandas.api.types module.
"""

import pandas.api.types as _types

from sktime.compat.pandas import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
)

__all__ = [
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_datetime64_any_dtype",
    "is_float_dtype",
    "is_integer_dtype",
    "is_numeric_dtype",
    "is_signed_integer_dtype",
    "is_string_dtype",
    "is_timedelta64_dtype",
    "is_unsigned_integer_dtype",
]


def __getattr__(name):
    return getattr(_types, name)
