"""Pandas version compatibility layer for sktime.

Provides a stable internal API for pandas functionality that differs across
pandas 1.x, 2.x, and 3.x. All version-specific pandas logic is concentrated
here so the rest of sktime can remain version-agnostic.

Import from ``sktime.compat.pandas`` instead of using version-dependent
pandas APIs directly.

Version support: pandas >= 1.1, < 4.0
"""
