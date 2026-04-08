"""Dtype compatibility for pandas v1/v2/v3.

In pandas v3, "silent downcasting" was removed: operations like fillna, where,
mask, and item assignment no longer silently change column dtypes.  Code that
relied on creating an all-NaN object-dtype DataFrame and then filling it with
numeric values now keeps the object dtype instead of becoming float64.

Additionally, pandas v3 returns nullable extension types (Int64, Float64,
boolean, string) more frequently from methods like convert_dtypes().

This module provides helpers to handle both issues.
"""

import numpy as np
import pandas as pd

_NULLABLE_NUMERIC_DTYPES = frozenset(
    [
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "boolean",
    ]
)


def is_nullable_numeric_dtype(dtype):
    """Check whether a dtype is a pandas nullable numeric or boolean ExtensionDtype.

    Covers all nullable integer types (Int8 through Int64, UInt8 through UInt64),
    nullable float types (Float32, Float64), and nullable boolean.
    Does NOT cover StringDtype or CategoricalDtype.

    Parameters
    ----------
    dtype : dtype-like
        A pandas dtype, numpy dtype, or its string representation.

    Returns
    -------
    bool
    """
    return str(dtype) in _NULLABLE_NUMERIC_DTYPES


def coerce_nullable_dtypes(obj):
    """Coerce nullable numeric columns to numpy float64.

    Drop-in replacement for ``_coerce_df_dtypes`` with extended type coverage.
    The original only handled Int64, Float64, and boolean; this version handles
    all nullable integer widths (Int8..Int64, UInt8..UInt64), both float widths,
    and boolean.

    Does not mutate the input.  Returns a shallow copy when coercion is needed.

    Parameters
    ----------
    obj : pd.Series, pd.DataFrame, or any object

    Returns
    -------
    Coerced copy if nullable columns were found, otherwise obj unchanged.
    """
    if isinstance(obj, pd.Series):
        if is_nullable_numeric_dtype(obj.dtype):
            return obj.astype("float64")
        return obj

    if isinstance(obj, pd.DataFrame):
        nullable_cols = [
            col for col in obj.columns if is_nullable_numeric_dtype(obj.dtypes[col])
        ]
        if nullable_cols:
            obj = obj.astype(dict.fromkeys(nullable_cols, "float64"))
        return obj

    return obj


def make_numeric_dataframe(index, columns, fill_value=np.nan, dtype="float64"):
    """Create a DataFrame pre-filled with a value and a guaranteed numeric dtype.

    This replaces the pattern that silently breaks in pandas v3::

        df = pd.DataFrame(index=idx, columns=cols)   # object dtype!
        df.fillna(0, inplace=True)                    # stays object in v3

    Use instead::

        df = make_numeric_dataframe(idx, cols, fill_value=0)

    Parameters
    ----------
    index : pd.Index
        Row index.
    columns : list-like
        Column names.
    fill_value : scalar, default np.nan
        Value to fill every cell with.
    dtype : str or np.dtype, default "float64"
        Desired dtype for all columns.

    Returns
    -------
    pd.DataFrame
    """
    data = np.full((len(index), len(columns)), fill_value)
    return pd.DataFrame(data, index=index, columns=columns, dtype=dtype)
