"""Unit tests for schema inference functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app import (
    ALIASES,
    SEMANTIC_ROLES,
    default_mapping,
    infer_schema,
    role_score,
)


class TestRoleScore:
    def test_exact_match_retailer(self):
        series = pd.Series(["Walmart", "Target", "Kroger"])
        score, reason = role_score(series, "Retailer", "retailer")
        assert score == 1.0
        assert reason == "Strong column-name match"

    def test_exact_match_brand(self):
        series = pd.Series(["Coca-Cola", "Pepsi", "Sprite"])
        score, reason = role_score(series, "Brand", "brand")
        assert score == 1.0

    def test_partial_match(self):
        series = pd.Series([1, 2, 3])
        score, reason = role_score(series, "Pack_Size", "pack_size")
        assert score >= 0.78
        assert reason in ("Strong column-name match", "Partial column-name match")

    def test_numeric_signal_for_quantity(self):
        series = pd.Series([100, 200, 300, 400, 500])
        score, reason = role_score(series, "Random_Col", "quantity")
        assert score >= 0.12
        assert reason == "Compatible data characteristics"

    def test_date_signal_for_period(self):
        series = pd.Series(pd.date_range("2024-01-01", periods=10, freq="MS"))
        score, reason = role_score(series, "Random_Col", "period")
        assert score >= 0.22
        assert reason == "Compatible data characteristics"

    def test_categorical_signal_for_retailer(self):
        series = pd.Series(["Walmart", "Target", "Kroger", "Walmart"])
        score, reason = role_score(series, "Random_Col", "retailer")
        assert score >= 0.12
        assert reason == "Compatible data characteristics"

    def test_low_score_for_mismatch(self):
        series = pd.Series(["a", "b", "c"])
        score, reason = role_score(series, "Random_Col", "quantity")
        assert score < 0.35
        assert reason == "Weak semantic evidence"


class TestInferSchema:
    def test_basic_inference(self):
        df = pd.DataFrame({
            "Retailer": ["Walmart", "Target"],
            "Brand": ["Coke", "Pepsi"],
            "Pack_Size": [2.0, 1.5],
            "Sales": [1000, 2000],
            "Qty": [100, 200],
            "Price": [10.0, 10.0],
            "Month": ["Jan 2024", "Feb 2024"],
        })
        result = infer_schema(df)
        assert len(result) == len(df.columns)
        assert "CSV column" in result.columns
        assert "Detected role" in result.columns
        assert "Confidence" in result.columns
        assert "Score" in result.columns

    def test_confidence_levels(self):
        df = pd.DataFrame({
            "Retailer": ["Walmart", "Target"],
            "Random_Col": ["a", "b"],
        })
        result = infer_schema(df)
        retailer_row = result[result["CSV column"] == "Retailer"].iloc[0]
        assert retailer_row["Confidence"] == "High"
        random_row = result[result["CSV column"] == "Random_Col"].iloc[0]
        assert random_row["Confidence"] == "Low"

    def test_unmapped_low_score(self):
        df = pd.DataFrame({"Unknown_Column": ["x", "y", "z"]})
        result = infer_schema(df)
        row = result.iloc[0]
        assert row["Detected role"] == "Unmapped"
        assert row["Confidence"] == "Low"


class TestDefaultMapping:
    def test_basic_mapping(self):
        inference = pd.DataFrame({
            "CSV column": ["Retailer", "Brand", "Pack_Size", "Sales", "Qty", "Price", "Month"],
            "Detected role": ["retailer", "brand", "pack_size", "sales", "quantity", "unit_price", "period"],
            "Score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        mapping = default_mapping(inference)
        assert mapping["retailer"] == "Retailer"
        assert mapping["brand"] == "Brand"
        assert mapping["pack_size"] == "Pack_Size"
        assert mapping["sales"] == "Sales"
        assert mapping["quantity"] == "Qty"
        assert mapping["unit_price"] == "Price"
        assert mapping["period"] == "Month"

    def test_no_duplicate_mappings(self):
        inference = pd.DataFrame({
            "CSV column": ["Col1", "Col2"],
            "Detected role": ["retailer", "retailer"],
            "Score": [0.9, 0.8],
        })
        mapping = default_mapping(inference)
        assert mapping["retailer"] == "Col1"
        assert mapping["brand"] is None

    def test_all_roles_present(self):
        inference = pd.DataFrame({
            "CSV column": [f"Col_{r}" for r in SEMANTIC_ROLES],
            "Detected role": SEMANTIC_ROLES,
            "Score": [1.0] * len(SEMANTIC_ROLES),
        })
        mapping = default_mapping(inference)
        for role in SEMANTIC_ROLES:
            assert mapping[role] == f"Col_{role}"


class TestAliases:
    def test_aliases_exist_for_all_roles(self):
        for role in SEMANTIC_ROLES:
            assert role in ALIASES
            assert len(ALIASES[role]) > 0
            # Self should be in aliases (normalized)
            normalized_role = role.replace("_", " ")
            assert normalized_role in ALIASES[role]

    def test_common_aliases(self):
        assert "chain" in ALIASES["retailer"]
        assert "customer" in ALIASES["retailer"]
        assert "revenue" in ALIASES["sales"]
        assert "qty" in ALIASES["quantity"]
        assert "price" in ALIASES["unit_price"]
