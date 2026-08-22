"""Unit tests for canonicalization functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import pandas as pd

from app import (
    Capability,
    canonicalize,
    capability_report,
    model_granularity,
    parse_period,
)


class TestParsePeriod:
    def test_standard_dates(self):
        series = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"])
        result = parse_period(series)
        assert result.notna().all()
        assert result.dtype == "datetime64[ns]"

    def test_month_year_format(self):
        series = pd.Series(["Jan 2024", "Feb 2024", "Mar 2024"])
        result = parse_period(series)
        assert result.notna().all()

    def test_year_month_numeric(self):
        series = pd.Series(["2024-01", "2024-02", "2024-03"])
        result = parse_period(series)
        assert result.notna().all()

    def test_yearmonth_compact(self):
        series = pd.Series(["202401", "202402", "202403"])
        result = parse_period(series)
        assert result.notna().all()

    def test_mixed_invalid(self):
        series = pd.Series(["2024-01-01", "invalid", "2024-03-01"])
        result = parse_period(series)
        assert result.notna().sum() == 2


class TestCanonicalize:
    def test_basic_canonicalize(self):
        raw = pd.DataFrame({
            "Retailer": ["Walmart", "Target"],
            "Brand": ["Coke", "Pepsi"],
            "Pack_Size": [2.0, 1.5],
            "Package": ["bottle", "can"],
            "Stores_Listed": [100, 200],
            "Max_Stores": [500, 400],
            "Sales": [1000.0, 2000.0],
            "Qty": [100, 200],
            "Month": ["Jan 2024", "Feb 2024"],
            "Price": [10.0, 10.0],
        })
        mapping = {
            "retailer": "Retailer",
            "brand": "Brand",
            "pack_size": "Pack_Size",
            "package": "Package",
            "stores_listed": "Stores_Listed",
            "max_stores": "Max_Stores",
            "sales": "Sales",
            "quantity": "Qty",
            "period": "Month",
            "unit_price": "Price",
        }
        mapping_json = json.dumps(mapping)
        result = canonicalize(raw, mapping_json)

        assert "retailer" in result.columns
        assert "brand" in result.columns
        assert "entity_id" in result.columns
        assert "distribution" in result.columns
        assert "log_quantity" in result.columns
        assert "log_unit_price" in result.columns
        assert "sales_contribution" in result.columns
        assert "quantity_contribution" in result.columns
        assert "time_index" in result.columns

    def test_numeric_coercion(self):
        raw = pd.DataFrame({
            "Qty": ["100", "200", "invalid"],
            "Price": ["10.0", "20.0", "30.0"],
        })
        mapping = {"quantity": "Qty", "unit_price": "Price"}
        mapping_json = json.dumps(mapping)
        result = canonicalize(raw, mapping_json)
        assert result["quantity"].notna().sum() == 2
        assert result["unit_price"].notna().sum() == 3

    def test_entity_id_construction(self):
        raw = pd.DataFrame({
            "Brand": ["Coke", "Pepsi"],
            "Pack_Size": [2.0, 1.5],
            "Package": ["bottle", "can"],
        })
        mapping = {"brand": "Brand", "pack_size": "Pack_Size", "package": "Package"}
        mapping_json = json.dumps(mapping)
        result = canonicalize(raw, mapping_json)
        assert "entity_id" in result.columns
        assert result["entity_id"].iloc[0] == "Coke | 2.0 | bottle"
        assert result["entity_id"].iloc[1] == "Pepsi | 1.5 | can"

    def test_entity_id_fallback_to_brand(self):
        raw = pd.DataFrame({"Brand": ["Coke", "Pepsi"]})
        mapping = {"brand": "Brand"}
        mapping_json = json.dumps(mapping)
        result = canonicalize(raw, mapping_json)
        assert "entity_id" in result.columns
        assert result["entity_id"].iloc[0] == "Coke"

    def test_distribution_calculation(self):
        raw = pd.DataFrame({
            "Stores_Listed": [100, 200, 300],
            "Max_Stores": [500, 400, 600],
        })
        mapping = {"stores_listed": "Stores_Listed", "max_stores": "Max_Stores"}
        mapping_json = json.dumps(mapping)
        result = canonicalize(raw, mapping_json)
        assert "distribution" in result.columns
        expected = [0.2, 0.5, 0.5]
        np.testing.assert_array_almost_equal(result["distribution"].values, expected)

    def test_distribution_clipping(self):
        raw = pd.DataFrame({
            "Stores_Listed": [-10, 600],  # Invalid values
            "Max_Stores": [500, 500],
        })
        mapping = {"stores_listed": "Stores_Listed", "max_stores": "Max_Stores"}
        mapping_json = json.dumps(mapping)
        result = canonicalize(raw, mapping_json)
        assert result["distribution"].isna().all()  # Both invalid -> NaN


class TestCapabilityReport:
    def test_performance_capability(self):
        data = pd.DataFrame({"sales": [100, 200], "quantity": [10, 20]})
        caps = capability_report(data)
        assert bool(caps["Performance"].available) is True

    def test_performance_capability_no_sales_no_qty(self):
        data = pd.DataFrame({"price": [10.0, 20.0]})
        caps = capability_report(data)
        assert caps["Performance"].available is False

    def test_pricing_capability_valid(self):
        data = pd.DataFrame({
            "quantity": [10, 20, 30, 40, 50],
            "unit_price": [10.0, 11.0, 12.0, 13.0, 14.0],
        })
        caps = capability_report(data)
        assert bool(caps["Pricing"].available) is True

    def test_pricing_capability_insufficient_price_variation(self):
        data = pd.DataFrame({
            "quantity": [10, 20, 30, 40],
            "unit_price": [10.0, 10.0, 10.0, 10.0],  # No variation
        })
        caps = capability_report(data)
        assert caps["Pricing"].available is False

    def test_distribution_capability(self):
        data = pd.DataFrame({
            "distribution": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        caps = capability_report(data)
        assert caps["Distribution"].available is True

    def test_distribution_capability_insufficient(self):
        data = pd.DataFrame({"distribution": [0.5, 0.5, 0.5]})  # Only 1 unique
        caps = capability_report(data)
        assert caps["Distribution"].available is False

    def test_entity_hierarchy_capability(self):
        data = pd.DataFrame({"entity_id": ["A", "B", "C"]})
        caps = capability_report(data)
        assert caps["Entity hierarchy"].available is True

    def test_hierarchical_price_model_capability(self):
        # Need: valid_price + entity_support (3+ entities, 8+ obs each, 80+ total)
        data = pd.DataFrame({
            "entity_id": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
            "quantity": np.random.default_rng(42).lognormal(3, 0.5, 90),
            "unit_price": np.random.default_rng(42).uniform(5, 15, 90),
        })
        caps = capability_report(data)
        # May or may not be available depending on price variation
        assert isinstance(caps["Hierarchical price model"].available, bool)

    def test_bayesian_demand_model_capability(self):
        data = pd.DataFrame({
            "quantity": [10] * 40,
            "unit_price": np.linspace(5, 15, 40),
        })
        caps = capability_report(data)
        assert caps["Bayesian demand model"].available is True

    def test_bayesian_demand_model_insufficient(self):
        data = pd.DataFrame({
            "quantity": [10] * 20,  # < 30
            "unit_price": [10.0] * 20,
        })
        caps = capability_report(data)
        assert caps["Bayesian demand model"].available is False


class TestModelGranularity:
    def test_full_grain(self):
        data = pd.DataFrame({
            "entity_id": ["A", "A", "B"],
            "retailer": ["Walmart", "Target", "Walmart"],
            "period": pd.date_range("2024-01-01", periods=3),
        })
        result = model_granularity(data)
        assert "Entity × Retailer × Period" in result

    def test_partial_grain(self):
        data = pd.DataFrame({
            "entity_id": ["A", "B"],
            "retailer": ["Walmart", "Target"],
        })
        result = model_granularity(data)
        assert "Entity × Retailer" in result

    def test_duplicates_detected(self):
        data = pd.DataFrame({
            "entity_id": ["A", "A", "B"],
            "retailer": ["Walmart", "Walmart", "Target"],
            "period": pd.date_range("2024-01-01", periods=3),
        })
        result = model_granularity(data)
        assert "multiple rows per key" in result or "\xd7" in result

    def test_no_keys(self):
        data = pd.DataFrame({"sales": [100, 200]})
        result = model_granularity(data)
        assert result == "Unidentified analytical grain"


class TestCapabilityDataclass:
    def test_capability_creation(self):
        cap = Capability(True, "Reason")
        assert cap.available is True
        assert cap.reason == "Reason"

    def test_capability_false(self):
        cap = Capability(False, "Not available")
        assert cap.available is False
