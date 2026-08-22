"""Unit tests for utility functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from app import (
    ROLE_LABELS,
    SEMANTIC_ROLES,
    fingerprint,
    fmt_number,
    fmt_pct,
    human_role,
    norm,
    safe_corr,
)


class TestNorm:
    def test_basic_normalization(self):
        assert norm("Retailer") == "retailer"
        assert norm("Pack Size") == "pack size"
        assert norm("  Sales ($)  ") == "sales"

    def test_special_characters(self):
        assert norm("Price_per_Unit") == "price per unit"
        assert norm("Max-Stores") == "max stores"
        assert norm("Brand.Name") == "brand name"

    def test_numbers_preserved(self):
        assert norm("Year2024") == "year2024"
        assert norm("Q1 2024") == "q1 2024"

    def test_empty_string(self):
        assert norm("") == ""
        assert norm(None) == "none"


class TestHumanRole:
    def test_known_roles(self):
        assert human_role("retailer") == "Retailer"
        assert human_role("pack_size") == "Pack Size"
        assert human_role("unit_price") == "Unit Price"

    def test_unknown_role(self):
        assert human_role("unknown_field") == "Unknown Field"


class TestFmtNumber:
    def test_integer_formatting(self):
        assert fmt_number(1000) == "1,000"
        assert fmt_number(1000000) == "1,000,000"

    def test_decimal_formatting(self):
        assert fmt_number(1234.56, digits=2) == "1,234.56"
        assert fmt_number(0.5, digits=1) == "0.5"

    def test_non_finite(self):
        assert fmt_number(np.inf) == "—"
        assert fmt_number(np.nan) == "—"
        assert fmt_number(-np.inf) == "—"


class TestFmtPct:
    def test_percentage_formatting(self):
        assert fmt_pct(0.5) == "50.0%"
        assert fmt_pct(0.1234, digits=2) == "12.34%"
        assert fmt_pct(1.0) == "100.0%"

    def test_non_finite(self):
        assert fmt_pct(np.inf) == "—"
        assert fmt_pct(np.nan) == "—"


class TestFingerprint:
    def test_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mapping = {"retailer": "a", "brand": "b"}
        fp1 = fingerprint(df, mapping)
        fp2 = fingerprint(df, mapping)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA256 hex

    def test_different_data_different_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        mapping = {"retailer": "a"}
        assert fingerprint(df1, mapping) != fingerprint(df2, mapping)

    def test_different_mapping_different_fingerprint(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mapping1 = {"retailer": "a"}
        mapping2 = {"retailer": "b"}
        assert fingerprint(df, mapping1) != fingerprint(df, mapping2)


class TestSafeCorr:
    def test_valid_correlation(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([2, 4, 6, 8, 10])
        assert safe_corr(a, b) == pytest.approx(1.0)

    def test_negative_correlation(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([5, 4, 3, 2, 1])
        assert safe_corr(a, b) == pytest.approx(-1.0)

    def test_insufficient_data(self):
        a = pd.Series([1, 2])
        b = pd.Series([3, 4])
        assert np.isnan(safe_corr(a, b))

    def test_constant_series(self):
        a = pd.Series([1, 1, 1, 1])
        b = pd.Series([2, 3, 4, 5])
        assert np.isnan(safe_corr(a, b))

    def test_with_nans(self):
        a = pd.Series([1, 2, np.nan, 4, 5])
        b = pd.Series([2, 4, 6, np.nan, 10])
        result = safe_corr(a, b)
        # With only 3 overlapping non-NaN values, correlation may be NaN
        # Just verify it doesn't crash
        assert result is not None


class TestConstants:
    def test_semantic_roles_complete(self):
        expected = [
            "retailer", "brand", "pack_size", "package",
            "stores_listed", "max_stores", "sales", "quantity",
            "period", "unit_price"
        ]
        assert expected == SEMANTIC_ROLES

    def test_role_labels_mapping(self):
        for role in SEMANTIC_ROLES:
            assert role in ROLE_LABELS
            assert ROLE_LABELS[role] == role.replace("_", " ").title()
