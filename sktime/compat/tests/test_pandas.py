"""Tests for sktime.compat.pandas compatibility layer.

These tests verify that every compat function produces correct results on the
installed pandas version. They do not branch on version: each assertion must
hold regardless of whether pandas 1.x, 2.x, or 3.x is installed. This way
the same test file validates the compat layer on all supported versions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.compat.pandas import (
    PANDAS_VERSION,
    df_map,
    ensure_index,
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_nested_object,
    is_numeric_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
    normalize_freq,
    offset_freq,
    period_freq,
    set_freq,
    suppress_freq_warning,
)


class TestPandasVersion:
    """Verify that version detection works correctly."""

    def test_version_is_int_tuple(self):
        assert isinstance(PANDAS_VERSION, tuple)
        assert len(PANDAS_VERSION) == 3
        assert all(isinstance(v, int) for v in PANDAS_VERSION)

    def test_version_matches_pandas(self):
        major, minor = PANDAS_VERSION[:2]
        parts = pd.__version__.split(".")
        assert major == int(parts[0])
        assert minor == int(parts[1])


class TestIsFloatDtype:
    @pytest.mark.parametrize(
        "inp",
        [np.float16, np.float32, np.float64, np.dtype("float32")],
    )
    def test_true_for_numpy_floats(self, inp):
        assert is_float_dtype(inp) is True

    @pytest.mark.parametrize(
        "inp",
        [np.int32, np.int64, np.uint8, np.bool_, np.dtype("U10"), np.dtype("O")],
    )
    def test_false_for_non_floats(self, inp):
        assert is_float_dtype(inp) is False

    def test_numpy_array(self):
        assert is_float_dtype(np.array([1.0, 2.0])) is True
        assert is_float_dtype(np.array([1, 2])) is False

    def test_pandas_series(self):
        assert is_float_dtype(pd.Series([1.0, 2.0])) is True
        assert is_float_dtype(pd.Series([1, 2])) is False

    def test_nullable_float(self):
        if hasattr(pd, "Float64Dtype"):
            assert is_float_dtype(pd.Float64Dtype()) is True

    def test_nullable_int_is_not_float(self):
        if hasattr(pd, "Int64Dtype"):
            assert is_float_dtype(pd.Int64Dtype()) is False


class TestIsIntegerDtype:
    @pytest.mark.parametrize(
        "inp",
        [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32, np.uint64],
    )
    def test_true_for_numpy_ints(self, inp):
        assert is_integer_dtype(inp) is True

    @pytest.mark.parametrize(
        "inp",
        [np.float64, np.bool_, np.dtype("U10"), np.dtype("datetime64[ns]")],
    )
    def test_false_for_non_ints(self, inp):
        assert is_integer_dtype(inp) is False

    def test_pandas_index(self):
        assert is_integer_dtype(pd.Index([1, 2, 3])) is True
        assert is_integer_dtype(pd.RangeIndex(10)) is True

    def test_datetime_index_is_not_int(self):
        assert is_integer_dtype(pd.DatetimeIndex(["2020-01-01"])) is False

    def test_nullable_int(self):
        if hasattr(pd, "Int64Dtype"):
            assert is_integer_dtype(pd.Int64Dtype()) is True
        if hasattr(pd, "UInt32Dtype"):
            assert is_integer_dtype(pd.UInt32Dtype()) is True


class TestIsSignedIntegerDtype:
    def test_signed(self):
        assert is_signed_integer_dtype(np.int64) is True
        assert is_signed_integer_dtype(np.int8) is True

    def test_unsigned_is_false(self):
        assert is_signed_integer_dtype(np.uint64) is False

    def test_float_is_false(self):
        assert is_signed_integer_dtype(np.float64) is False

    def test_nullable_signed(self):
        if hasattr(pd, "Int64Dtype"):
            assert is_signed_integer_dtype(pd.Int64Dtype()) is True

    def test_nullable_unsigned_is_false(self):
        if hasattr(pd, "UInt64Dtype"):
            assert is_signed_integer_dtype(pd.UInt64Dtype()) is False


class TestIsUnsignedIntegerDtype:
    def test_unsigned(self):
        assert is_unsigned_integer_dtype(np.uint64) is True
        assert is_unsigned_integer_dtype(np.uint8) is True

    def test_signed_is_false(self):
        assert is_unsigned_integer_dtype(np.int64) is False

    def test_nullable_unsigned(self):
        if hasattr(pd, "UInt64Dtype"):
            assert is_unsigned_integer_dtype(pd.UInt64Dtype()) is True


class TestIsBoolDtype:
    def test_numpy_bool(self):
        assert is_bool_dtype(np.bool_) is True
        assert is_bool_dtype(np.dtype("bool")) is True

    def test_int_is_not_bool(self):
        assert is_bool_dtype(np.int64) is False

    def test_series(self):
        assert is_bool_dtype(pd.Series([True, False])) is True
        assert is_bool_dtype(pd.Series([1, 0])) is False

    def test_nullable_bool(self):
        if hasattr(pd, "BooleanDtype"):
            assert is_bool_dtype(pd.BooleanDtype()) is True


class TestIsStringDtype:
    def test_numpy_object(self):
        assert is_string_dtype(np.dtype("O")) is True

    def test_numpy_unicode(self):
        assert is_string_dtype(np.dtype("U10")) is True

    def test_numpy_bytes(self):
        assert is_string_dtype(np.dtype("S10")) is True

    def test_numeric_is_false(self):
        assert is_string_dtype(np.int64) is False
        assert is_string_dtype(np.float64) is False

    def test_categorical_is_false(self):
        assert is_string_dtype(pd.CategoricalDtype()) is False

    def test_pandas_string_dtype(self):
        if hasattr(pd, "StringDtype"):
            assert is_string_dtype(pd.StringDtype()) is True

    def test_series_of_strings(self):
        s = pd.Series(["a", "b", "c"], dtype=object)
        assert is_string_dtype(s) is True


class TestIsNumericDtype:
    @pytest.mark.parametrize(
        "inp",
        [np.float64, np.int32, np.uint8, np.complex128],
    )
    def test_true_for_numeric(self, inp):
        assert is_numeric_dtype(inp) is True

    @pytest.mark.parametrize(
        "inp",
        [np.bool_, np.dtype("O"), np.dtype("U10"), np.dtype("datetime64[ns]")],
    )
    def test_false_for_non_numeric(self, inp):
        assert is_numeric_dtype(inp) is False

    def test_nullable_numeric(self):
        if hasattr(pd, "Int64Dtype"):
            assert is_numeric_dtype(pd.Int64Dtype()) is True
        if hasattr(pd, "Float64Dtype"):
            assert is_numeric_dtype(pd.Float64Dtype()) is True


class TestIsDatetime64AnyDtype:
    def test_numpy_datetime(self):
        assert is_datetime64_any_dtype(np.dtype("datetime64[ns]")) is True
        assert is_datetime64_any_dtype(np.dtype("datetime64[us]")) is True

    def test_non_datetime(self):
        assert is_datetime64_any_dtype(np.int64) is False
        assert is_datetime64_any_dtype(np.dtype("timedelta64[ns]")) is False

    def test_datetime_index(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
        assert is_datetime64_any_dtype(idx) is True

    def test_tz_aware(self):
        idx = pd.DatetimeIndex(["2020-01-01"], tz="UTC")
        assert is_datetime64_any_dtype(idx) is True


class TestIsTimedelta64Dtype:
    def test_timedelta(self):
        assert is_timedelta64_dtype(np.dtype("timedelta64[ns]")) is True

    def test_non_timedelta(self):
        assert is_timedelta64_dtype(np.int64) is False
        assert is_timedelta64_dtype(np.dtype("datetime64[ns]")) is False

    def test_timedelta_index(self):
        idx = pd.TimedeltaIndex(["1 day", "2 days"])
        assert is_timedelta64_dtype(idx) is True


class TestIsCategoricalDtype:
    def test_categorical_dtype(self):
        assert is_categorical_dtype(pd.CategoricalDtype()) is True

    def test_non_categorical(self):
        assert is_categorical_dtype(np.dtype("O")) is False
        assert is_categorical_dtype(np.int64) is False

    def test_categorical_array(self):
        cat = pd.Categorical(["a", "b", "c"])
        assert is_categorical_dtype(cat) is True

    def test_categorical_series(self):
        s = pd.Series(pd.Categorical(["x", "y"]))
        assert is_categorical_dtype(s) is True


class TestOffsetFreq:
    """Verify that offset_freq returns working frequency strings.

    Each test creates an actual pandas object with the returned freq to
    confirm it is valid on the installed version.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "month_end",
            "quarter_end",
            "year_end",
            "hour",
            "minute",
            "second",
            "day",
            "week",
            "business_day",
            "month_start",
        ],
    )
    def test_offset_produces_valid_date_range(self, name):
        freq = offset_freq(name)
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_business_month_end(self):
        freq = offset_freq("business_month_end")
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_year_start(self):
        freq = offset_freq("year_start")
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown offset"):
            offset_freq("does_not_exist")


class TestPeriodFreq:
    """Verify that period_freq returns working period frequency strings."""

    @pytest.mark.parametrize(
        "name, start",
        [
            ("year", "2020"),
            ("quarter", "2020Q1"),
            ("month", "2020-01"),
            ("week", "2020-01-06"),
            ("day", "2020-01-01"),
            ("hour", "2020-01-01 00:00"),
            ("minute", "2020-01-01 00:00"),
            ("second", "2020-01-01 00:00:00"),
        ],
    )
    def test_period_produces_valid_range(self, name, start):
        freq = period_freq(name)
        idx = pd.period_range(start, periods=3, freq=freq)
        assert len(idx) == 3

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown period"):
            period_freq("does_not_exist")


class TestNormalizeFreq:
    """Verify frequency normalization produces valid strings."""

    def test_none_passthrough(self):
        assert normalize_freq(None) is None

    def test_non_string_passthrough(self):
        offset = pd.tseries.frequencies.to_offset(offset_freq("day"))
        assert normalize_freq(offset) is offset

    @pytest.mark.parametrize("freq_str", ["D", "W", "B"])
    def test_unchanged_aliases(self, freq_str):
        assert normalize_freq(freq_str) == freq_str

    def test_compound_unchanged(self):
        assert normalize_freq("2D") == "2D"
        assert normalize_freq("3W") == "3W"

    @pytest.mark.parametrize(
        "old_or_new",
        ["M", "ME", "H", "h", "T", "min", "Q", "QE", "Y", "YE", "S", "s"],
    )
    def test_normalized_freq_works_with_date_range(self, old_or_new):
        freq = normalize_freq(old_or_new)
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_compound_normalization(self):
        freq = normalize_freq("2M")
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_anchored_normalization(self):
        freq = normalize_freq("Q-FEB")
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_anchored_new_style(self):
        freq = normalize_freq("QE-FEB")
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_year_anchored(self):
        for alias in ["Y-JAN", "YE-JAN", "A-JAN"]:
            freq = normalize_freq(alias)
            idx = pd.date_range("2020-01-01", periods=3, freq=freq)
            assert len(idx) == 3

    def test_compound_with_anchor(self):
        freq = normalize_freq("2Q-MAR")
        idx = pd.date_range("2020-01-01", periods=3, freq=freq)
        assert len(idx) == 3

    def test_empty_string(self):
        assert normalize_freq("") == ""


class TestDfMap:
    def test_returns_callable(self):
        df = pd.DataFrame({"a": [1]})
        assert callable(df_map(df))

    def test_element_wise_operation(self):
        df = pd.DataFrame({"a": [1, 4], "b": [9, 16]})
        result = df_map(df)(np.sqrt)
        assert result.iloc[0, 0] == 1.0
        assert result.iloc[0, 1] == 3.0
        assert result.iloc[1, 0] == 2.0
        assert result.iloc[1, 1] == 4.0

    def test_string_conversion(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = df_map(df)(str)
        assert result.iloc[0, 0] == "1"
        assert result.iloc[1, 0] == "2"


class TestSetFreq:
    def test_datetime_index(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
        result = set_freq(idx, "D")
        assert result.freq is not None
        assert len(result) == 3

    def test_period_index_already_has_freq(self):
        freq_str = period_freq("month")
        idx = pd.period_range("2020-01", periods=3, freq=freq_str)
        # PeriodIndex freq is read-only (set at creation), so just verify
        # that period_freq produced a valid freq for period_range
        assert idx.freq is not None
        assert len(idx) == 3

    def test_none_clears_freq(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        assert idx.freq is not None
        result = set_freq(idx, None)
        assert result.freq is None

    def test_offset_freq_string(self):
        freq = offset_freq("day")
        idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=3, freq=freq))
        result = set_freq(idx, freq)
        assert result.freq is not None

    def test_multiindex_level(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        arrays = [["a", "a", "a"], dates]
        mi = pd.MultiIndex.from_arrays(arrays, names=["group", "date"])

        result = set_freq(mi, "D", level=-1)
        assert isinstance(result, pd.MultiIndex)
        assert result.levels[-1].freq is not None


class TestEnsureIndex:
    def test_list_to_index(self):
        result = ensure_index([1, 2, 3])
        assert isinstance(result, pd.Index)
        assert list(result) == [1, 2, 3]

    def test_tuple_to_index(self):
        result = ensure_index((4, 5))
        assert isinstance(result, pd.Index)
        assert len(result) == 2

    def test_index_passthrough(self):
        idx = pd.Index([1, 2, 3])
        result = ensure_index(idx)
        assert result is idx

    def test_range_index_passthrough(self):
        idx = pd.RangeIndex(5)
        result = ensure_index(idx)
        assert result is idx

    def test_numpy_array(self):
        arr = np.array([10, 20, 30])
        result = ensure_index(arr)
        assert isinstance(result, pd.Index)
        assert len(result) == 3


class TestIsNestedObject:
    def test_nested_series(self):
        inner1 = pd.Series([1, 2, 3])
        inner2 = pd.Series([4, 5, 6])
        outer = pd.Series([inner1, inner2], dtype=object)
        assert is_nested_object(outer) is True

    def test_flat_series_int(self):
        assert is_nested_object(pd.Series([1, 2, 3])) is False

    def test_flat_series_string(self):
        assert is_nested_object(pd.Series(["a", "b"], dtype=object)) is False

    def test_empty_series(self):
        assert is_nested_object(pd.Series([], dtype=object)) is False

    def test_non_series(self):
        assert is_nested_object([1, 2, 3]) is False
        assert is_nested_object(pd.DataFrame({"a": [1]})) is False

    def test_object_dtype_with_none(self):
        s = pd.Series([None, None], dtype=object)
        assert is_nested_object(s) is False


class TestSuppressFreqWarning:
    def test_context_manager_runs(self):
        with suppress_freq_warning():
            pass

    def test_pandas_operations_inside(self):
        with suppress_freq_warning():
            idx = pd.date_range("2020", periods=3, freq=offset_freq("day"))
            assert len(idx) == 3

    def test_no_freq_warning_leaks(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            freq = offset_freq("month_end")
            with suppress_freq_warning():
                pd.date_range("2020", periods=3, freq=freq)
            freq_warns = [
                w
                for w in caught
                if issubclass(w.category, FutureWarning)
                and "deprecated" in str(w.message).lower()
            ]
            assert len(freq_warns) == 0, f"Unexpected warnings: {freq_warns}"
