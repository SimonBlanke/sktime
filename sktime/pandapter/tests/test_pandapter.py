"""Tests for the pandapter compatibility module.

Every test must pass regardless of the installed pandas version (1.x, 2.x, 3.x).
Tests validate behaviour, not version-specific strings.
"""

import numpy as np
import pandas as _pd

from sktime import pandapter as pd
from sktime.pandapter import (
    CompatDataFrame,
    CompatDatetimeIndex,
    CompatSeries,
    ensure_compat,
    ensure_native,
)


class TestCompatDataFrame:
    """CompatDataFrame subclass behavior."""

    def test_construction(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        assert type(df) is CompatDataFrame
        assert len(df) == 3

    def test_isinstance_real_pandas(self):
        df = pd.DataFrame({"a": [1]})
        assert isinstance(df, _pd.DataFrame)

    def test_isinstance_plain_df_passes_pandapter_check(self):
        real_df = _pd.DataFrame({"a": [1]})
        assert isinstance(real_df, pd.DataFrame)

    def test_constructor_preserves_type(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df + 1
        assert type(result) is CompatDataFrame

    def test_constructor_sliced_returns_compat_series(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        s = df["a"]
        assert type(s) is CompatSeries

    def test_groupby_preserves_type(self):
        df = pd.DataFrame({"g": ["x", "x", "y"], "v": [1, 2, 3]})
        result = df.groupby("g").sum()
        assert type(result) is CompatDataFrame

    def test_applymap(self):
        df = pd.DataFrame({"a": [1, 4], "b": [9, 16]})
        result = df.applymap(np.sqrt)
        assert result.iloc[0, 0] == 1.0
        assert result.iloc[1, 1] == 4.0
        assert type(result) is CompatDataFrame

    def test_resample_normalizes_freq(self):
        idx = pd.date_range("2020-01-01", periods=60, freq=pd.offsets.Day())
        df = pd.DataFrame({"v": range(60)}, index=idx)
        result = df.resample("D").mean()
        assert len(result) == 60


class TestCompatSeries:
    def test_construction(self):
        s = pd.Series([1, 2, 3], name="x")
        assert type(s) is CompatSeries

    def test_isinstance_real_pandas(self):
        s = pd.Series([1])
        assert isinstance(s, _pd.Series)

    def test_isinstance_plain_series_passes_pandapter_check(self):
        real_s = _pd.Series([1])
        assert isinstance(real_s, pd.Series)

    def test_constructor_preserves_type(self):
        s = pd.Series([1, 2, 3])
        result = s + 10
        assert type(result) is CompatSeries

    def test_constructor_expanddim(self):
        s = pd.Series([1, 2], name="col")
        result = s.to_frame()
        assert type(result) is CompatDataFrame

    def test_resample(self):
        idx = pd.date_range("2020-01-01", periods=30, freq=pd.offsets.Day())
        s = pd.Series(range(30), index=idx)
        result = s.resample("D").mean()
        assert len(result) == 30


class TestCompatDatetimeIndex:
    def test_freq_setter_normalizes(self):
        idx = _pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
        idx.__class__ = CompatDatetimeIndex
        idx.freq = "D"
        assert idx.freq is not None

    def test_freq_getter(self):
        idx = pd.date_range("2020-01-01", periods=3, freq=pd.offsets.Day())
        assert idx.freq is not None


class TestEnsureCompat:
    def test_converts_dataframe(self):
        df = _pd.DataFrame({"a": [1, 2]})
        result = ensure_compat(df)
        assert type(result) is CompatDataFrame
        assert result is df

    def test_converts_series(self):
        s = _pd.Series([1, 2])
        result = ensure_compat(s)
        assert type(result) is CompatSeries
        assert result is s

    def test_noop_on_compat_dataframe(self):
        df = CompatDataFrame({"a": [1]})
        result = ensure_compat(df)
        assert type(result) is CompatDataFrame

    def test_noop_on_none(self):
        assert ensure_compat(None) is None

    def test_converts_datetime_index(self):
        idx = _pd.date_range("2020-01-01", periods=3, freq="D")
        df = _pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        ensure_compat(df)
        assert type(df.index) is CompatDatetimeIndex

    def test_converts_multiindex_datetime_level(self):
        dates = _pd.date_range("2020-01-01", periods=3, freq="D")
        arrays = [["a", "a", "a"], dates]
        mi = _pd.MultiIndex.from_arrays(arrays, names=["group", "date"])
        df = _pd.DataFrame({"v": [1, 2, 3]}, index=mi)
        ensure_compat(df)
        assert type(df.index.levels[-1]) is CompatDatetimeIndex

    def test_data_preserved(self):
        df = _pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        ensure_compat(df)
        assert list(df["a"]) == [10, 20]
        assert list(df["b"]) == [30, 40]


class TestEnsureNative:
    def test_converts_back_dataframe(self):
        df = CompatDataFrame({"a": [1, 2]})
        result = ensure_native(df)
        assert type(result) is _pd.DataFrame
        assert result is df

    def test_converts_back_series(self):
        s = CompatSeries([1, 2])
        result = ensure_native(s)
        assert type(result) is _pd.Series
        assert result is s

    def test_noop_on_plain_pandas(self):
        df = _pd.DataFrame({"a": [1]})
        result = ensure_native(df)
        assert type(result) is _pd.DataFrame

    def test_noop_on_none(self):
        assert ensure_native(None) is None

    def test_reverts_datetime_index(self):
        idx = _pd.date_range("2020-01-01", periods=3, freq="D")
        df = _pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        ensure_compat(df)
        assert type(df.index) is CompatDatetimeIndex
        ensure_native(df)
        assert type(df.index) is _pd.DatetimeIndex

    def test_roundtrip_preserves_data(self):
        original = _pd.DataFrame({"x": [1.5, 2.5], "y": ["a", "b"]})
        ensure_compat(original)
        ensure_native(original)
        assert type(original) is _pd.DataFrame
        assert list(original["x"]) == [1.5, 2.5]
        assert list(original["y"]) == ["a", "b"]


class TestModuleProxy:
    """pandapter module-level API forwarding."""

    def test_date_range_normalizes_freq(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        assert len(idx) == 3

    def test_date_range_returns_compat_index(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        assert type(idx) is CompatDatetimeIndex

    def test_period_range(self):
        idx = pd.period_range("2020-01", periods=3, freq="M")
        assert len(idx) == 3

    def test_concat_returns_compat(self):
        df1 = _pd.DataFrame({"a": [1]})
        df2 = _pd.DataFrame({"a": [2]})
        result = pd.concat([df1, df2])
        assert type(result) is CompatDataFrame
        assert len(result) == 2

    def test_merge_returns_compat(self):
        left = _pd.DataFrame({"k": [1, 2], "v": [10, 20]})
        right = _pd.DataFrame({"k": [1, 2], "w": [30, 40]})
        result = pd.merge(left, right, on="k")
        assert type(result) is CompatDataFrame
        assert len(result) == 2

    def test_getattr_fallback(self):
        assert pd.NaT is _pd.NaT
        assert pd.Timestamp is _pd.Timestamp

    def test_to_datetime(self):
        result = pd.to_datetime(["2020-01-01"])
        assert len(result) == 1

    def test_isna(self):
        assert pd.isna(None) is True

    def test_offsets_available(self):
        assert pd.offsets.Day is _pd.offsets.Day


class TestApiTypes:
    """pandapter.api.types provides our compat functions."""

    def test_is_float_dtype(self):
        assert pd.api.types.is_float_dtype(np.float64) is True
        assert pd.api.types.is_float_dtype(np.int64) is False

    def test_is_integer_dtype(self):
        assert pd.api.types.is_integer_dtype(np.int32) is True
        assert pd.api.types.is_integer_dtype(np.float64) is False

    def test_is_bool_dtype(self):
        assert pd.api.types.is_bool_dtype(np.bool_) is True

    def test_is_string_dtype(self):
        assert pd.api.types.is_string_dtype(np.dtype("O")) is True

    def test_is_datetime64_any_dtype(self):
        assert pd.api.types.is_datetime64_any_dtype(np.dtype("datetime64[ns]")) is True

    def test_is_numeric_dtype(self):
        assert pd.api.types.is_numeric_dtype(np.float64) is True
        assert pd.api.types.is_numeric_dtype(np.bool_) is False

    def test_fallback_to_real_pandas(self):
        # pandas_dtype is not overridden, should come from real pandas
        result = pd.api.types.pandas_dtype("int64")
        assert result == np.dtype("int64")


class TestEndToEnd:
    """Simulate a simplified sktime data flow."""

    def test_user_data_through_compat_cycle(self):
        user_df = _pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=_pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        assert type(user_df) is _pd.DataFrame

        internal = ensure_compat(user_df)
        assert type(internal) is CompatDataFrame
        assert type(internal.index) is CompatDatetimeIndex

        result = internal + 10
        assert type(result) is CompatDataFrame
        assert list(result["value"]) == [11.0, 12.0, 13.0]

        output = ensure_native(result)
        assert type(output) is _pd.DataFrame
        assert type(output.index) is _pd.DatetimeIndex

    def test_internal_creation_has_compat_methods(self):
        df = pd.DataFrame({"a": [1, 4], "b": [9, 16]})
        result = df.applymap(np.sqrt)
        assert result.iloc[0, 0] == 1.0
        assert result.iloc[1, 1] == 4.0

    def test_series_operations_chain(self):
        s = pd.Series([1, 2, 3, 4])
        doubled = s * 2
        assert type(doubled) is CompatSeries
        df = doubled.to_frame("val")
        assert type(df) is CompatDataFrame
        assert list(df["val"]) == [2, 4, 6, 8]
