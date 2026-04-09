"""Coercion utilities for mtypes."""

__author__ = ["fkiraly"]

from sktime.utils.pandas_compat import coerce_nullable_dtypes, is_nullable_numeric_dtype

# keep old names for backwards compatibility with existing imports
_is_nullable_numeric = is_nullable_numeric_dtype
_coerce_df_dtypes = coerce_nullable_dtypes
