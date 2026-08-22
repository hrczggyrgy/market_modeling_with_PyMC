"""Unit tests for validation functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app import (
    SEMANTIC_ROLES,
    Finding,
    validate_mapping,
)


class TestValidateMapping:
    def test_empty_dataset_blocking(self):
        raw = pd.DataFrame()
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        findings = validate_mapping(raw, mapping)
        assert len(findings) == 1
        assert findings[0].severity == "Blocking"
        assert "Empty dataset" in findings[0].issue

    def test_duplicate_columns_blocking(self):
        raw = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        raw.columns = ["A", "A"]  # Duplicate column names
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "Duplicate column names" in f.issue for f in findings)

    def test_duplicate_role_mapping_blocking(self):
        raw = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["retailer"] = "Col1"
        mapping["brand"] = "Col1"  # Same column for two roles
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "One column mapped to multiple roles" in f.issue for f in findings)

    def test_missing_quantity_blocking(self):
        raw = pd.DataFrame({"Price": [10.0, 20.0], "Sales": [100, 200]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["unit_price"] = "Price"
        mapping["sales"] = "Sales"
        # quantity not mapped
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "Missing Quantity" in f.issue for f in findings)

    def test_missing_unit_price_blocking(self):
        raw = pd.DataFrame({"Qty": [10, 20], "Sales": [100, 200]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["sales"] = "Sales"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "Missing Unit Price" in f.issue for f in findings)

    def test_invalid_quantity_values_blocking(self):
        raw = pd.DataFrame({"Qty": [10, -5, 20], "Price": [10.0, 10.0, 10.0]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["unit_price"] = "Price"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "Non-positive Quantity" in f.issue for f in findings)

    def test_invalid_price_values_blocking(self):
        raw = pd.DataFrame({"Qty": [10, 20, 30], "Price": [10.0, 0.0, 10.0]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["unit_price"] = "Price"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "Non-positive Unit Price" in f.issue for f in findings)

    def test_insufficient_price_variation_blocking(self):
        raw = pd.DataFrame({"Qty": [10, 20, 30, 40], "Price": [10.0, 10.0, 10.0, 10.0]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["unit_price"] = "Price"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Blocking" and "Insufficient price variation" in f.issue for f in findings)

    def test_small_sample_warning(self):
        raw = pd.DataFrame({"Qty": [10, 20], "Price": [10.0, 20.0]})
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["unit_price"] = "Price"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Warning" and "Small sample" in f.issue for f in findings)

    def test_weak_period_parsing_warning(self):
        raw = pd.DataFrame({
            "Qty": [10, 20, 30],
            "Price": [10.0, 20.0, 30.0],
            "Period": ["invalid", "also_invalid", "still_bad"]
        })
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["unit_price"] = "Price"
        mapping["period"] = "Period"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Warning" and "Weak period parsing" in f.issue for f in findings)

    def test_invalid_store_coverage_warning(self):
        raw = pd.DataFrame({
            "Qty": [10, 20, 30],
            "Price": [10.0, 20.0, 30.0],
            "Stores_Listed": [100, 200, 300],
            "Max_Stores": [50, 100, 150],  # Stores > Max for first row
        })
        mapping = dict.fromkeys(SEMANTIC_ROLES)
        mapping["quantity"] = "Qty"
        mapping["unit_price"] = "Price"
        mapping["stores_listed"] = "Stores_Listed"
        mapping["max_stores"] = "Max_Stores"
        findings = validate_mapping(raw, mapping)
        assert any(f.severity == "Warning" and "Invalid store coverage" in f.issue for f in findings)

    def test_valid_mapping_passes(self):
        raw = pd.DataFrame({
            "Retailer": ["Walmart", "Target", "Kroger", "Walmart"],
            "Brand": ["Coke", "Pepsi", "Sprite", "Coke"],
            "Pack_Size": [2.0, 1.5, 1.0, 2.0],
            "Package": ["bottle", "can", "bottle", "bottle"],
            "Stores_Listed": [100, 200, 150, 100],
            "Max_Stores": [500, 400, 300, 500],
            "Sales": [1000, 2000, 1500, 1200],
            "Qty": [100, 200, 150, 120],
            "Month": ["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024"],
            "Price": [10.0, 11.0, 12.0, 13.0],  # 4 distinct prices
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
        findings = validate_mapping(raw, mapping)
        blocking = [f for f in findings if f.severity == "Blocking"]
        assert len(blocking) == 0


class TestFinding:
    def test_finding_creation(self):
        finding = Finding("Blocking", "Test Issue", "Test Detail")
        assert finding.severity == "Blocking"
        assert finding.issue == "Test Issue"
        assert finding.detail == "Test Detail"

    def test_finding_asdict(self):
        finding = Finding("Warning", "Test", "Detail")
        d = finding.__dict__
        assert d["severity"] == "Warning"
        assert d["issue"] == "Test"
        assert d["detail"] == "Detail"
