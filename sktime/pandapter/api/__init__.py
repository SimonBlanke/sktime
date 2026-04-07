"""Proxy for pandas.api with version-compatible type checking functions."""

import pandas.api as _api

from . import types  # noqa: F401 -- ensure pd.api.types resolves to our submodule


def __getattr__(name):
    return getattr(_api, name)
