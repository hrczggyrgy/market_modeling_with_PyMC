from __future__ import annotations

import hashlib
import io
import json
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymc as pm
import streamlit as st

# =============================================================================
# 01. CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Market Modeling",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "3.0.0"
RANDOM_SEED = 2026

SEMANTIC_ROLES = [
    "retailer",
    "brand",
    "pack_size",
    "package",
    "stores_listed",
    "max_stores",
    "sales",
    "quantity",
    "period",
    "unit_price",
]

ROLE_LABELS = {r: r.replace("_", " ").title() for r in SEMANTIC_ROLES}

ALIASES = {
    "retailer": ["retailer", "chain", "customer", "account", "banner"],
    "brand": ["brand", "manufacturer brand", "brand name"],
    "pack_size": ["pack size", "size", "volume", "net content", "packsize"],
    "package": ["package", "format", "container", "pack type", "packaging"],
    "stores_listed": [
        "stores listed",
        "listed stores",
        "distribution stores",
        "stores",
        "store count",
        "numeric distribution",
    ],
    "max_stores": [
        "max stores",
        "maximum stores",
        "total stores",
        "universe stores",
        "max stores retailer",
    ],
    "sales": ["sales", "sales $", "revenue", "net sales", "turnover", "value sales"],
    "quantity": ["qty", "quantity", "units", "volume", "unit sales", "demand"],
    "period": [
        "date",
        "month",
        "week",
        "period",
        "yearmonth",
        "year month",
        "quarter",
        "day",
    ],
    "unit_price": ["price", "unit price", "price per unit", "price_per_unit", "asp", "selling price"],
}


@dataclass
class Finding:
    severity: str
    issue: str
    detail: str


@dataclass
class Capability:
    available: bool
    reason: str


# =============================================================================
# 02. GENERAL UTILITIES
# =============================================================================


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def human_role(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " ").title())


def fmt_number(value: float, digits: int = 0) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}%}"


def fingerprint(df: pd.DataFrame, mapping: dict[str, str | None]) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(json.dumps(mapping, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 4 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# =============================================================================
# 03. SAMPLE DATA WITH KNOWN GROUND TRUTH
# =============================================================================


def make_synthetic_sample(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create a realistic research sample and return its known price elasticities."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-01-01", periods=36, freq="MS")

    truth = {
        "Coca-Cola | 2.0 | l": -1.10,
        "Coca-Cola | 0.5 | l": -0.95,
        "Pepsi | 1.5 | l": -0.90,
        "Sprite | 0.33 | l": -1.25,
        "Dr Pepper | 0.5 | l": -0.75,
        "Fanta | 2.0 | l": -1.00,
        "Mountain Dew | 1.25 | l": -1.15,
    }

    retailer_meta = {
        "Walmart": {"max_stores": 4700, "effect": 0.15},
        "Target": {"max_stores": 1950, "effect": -0.08},
        "Kroger": {"max_stores": 1240, "effect": -0.02},
    }
    entity_meta = {
        "Coca-Cola | 2.0 | l": (2.0, 11.6),
        "Coca-Cola | 0.5 | l": (0.5, 10.8),
        "Pepsi | 1.5 | l": (1.5, 11.2),
        "Sprite | 0.33 | l": (0.33, 10.5),
        "Dr Pepper | 0.5 | l": (0.5, 10.9),
        "Fanta | 2.0 | l": (2.0, 11.0),
        "Mountain Dew | 1.25 | l": (1.25, 11.15),
    }

    rows: list[list[Any]] = []
    for entity, epsilon in truth.items():
        brand, size, package = entity.split(" | ")
        pack_size = float(size)
        base = entity_meta[entity][1]
        for retailer, rmeta in retailer_meta.items():
            retailer_price_bias = {"Walmart": -0.03, "Target": 0.02, "Kroger": 0.01}[retailer]
            for t, dt in enumerate(months):
                seasonal = 0.10 * np.sin(2 * np.pi * dt.month / 12.0)
                promo_like = 0.06 * np.sin(2 * np.pi * (dt.month + 1) / 6.0)
                price = max(
                    0.40,
                    (0.95 + 0.72 * pack_size)
                    * (1 + retailer_price_bias)
                    * (1 + 0.035 * np.sin(t / 5.0) + rng.normal(0, 0.025))
                )
                distribution = np.clip(
                    0.74
                    + 0.11 * np.sin(t / 7.0)
                    + 0.025 * (retailer == "Walmart")
                    + rng.normal(0, 0.022),
                    0.35,
                    0.985,
                )
                log_q = (
                    base
                    + rmeta["effect"]
                    + epsilon * np.log(price)
                    + 0.38 * distribution
                    + seasonal
                    + promo_like
                    + 0.004 * t
                    + rng.normal(0, 0.10)
                )
                quantity = float(np.exp(log_q))
                sales = quantity * price
                rows.append(
                    [
                        retailer,
                        brand,
                        pack_size,
                        package,
                        round(rmeta["max_stores"] * distribution),
                        rmeta["max_stores"],
                        sales,
                        quantity,
                        dt.strftime("%b %Y"),
                        price,
                    ]
                )

    sample = pd.DataFrame(
        rows,
        columns=[
            "Retailer",
            "Brand",
            "Pack Size",
            "Package",
            "Stores Listed",
            "Max Stores (Retailer)",
            "Sales ($)",
            "Qty",
            "Month",
            "Price_per_Unit",
        ],
    )
    return sample, truth


@st.cache_data(show_spinner=False)
def sample_dataset() -> tuple[pd.DataFrame, dict[str, float]]:
    return make_synthetic_sample()


@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


# =============================================================================
# 04. SCHEMA INFERENCE
# =============================================================================


def role_score(series: pd.Series, column: str, role: str) -> tuple[float, str]:
    name = norm(column)
    aliases = [norm(a) for a in ALIASES[role]]
    exact = max((1.0 if name == alias else 0.0 for alias in aliases), default=0.0)
    partial = max((0.78 if alias in name or name in alias else 0.0 for alias in aliases), default=0.0)
    name_score = max(exact, partial)

    numeric_share = pd.to_numeric(series, errors="coerce").notna().mean()
    numeric = numeric_share >= 0.80
    positive_share = pd.to_numeric(series, errors="coerce").gt(0).mean() if numeric else 0.0
    date_share = pd.to_datetime(series, errors="coerce").notna().mean()

    signal = 0.0
    if role == "period" and date_share >= 0.60:
        signal = 0.22
    elif role in {"sales", "quantity", "unit_price", "stores_listed", "max_stores", "pack_size"} and numeric:
        signal = 0.12 if positive_share > 0.50 else 0.03
    elif role in {"retailer", "brand", "package"} and not numeric and series.nunique(dropna=True) > 1:
        signal = 0.12

    score = min(1.0, max(name_score, name_score * 0.78) + signal)
    if exact:
        reason = "Strong column-name match"
    elif partial:
        reason = "Partial column-name match"
    elif signal:
        reason = "Compatible data characteristics"
    else:
        reason = "Weak semantic evidence"

    return score, reason


def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        candidates = []
        for role in SEMANTIC_ROLES:
            score, reason = role_score(df[column], column, role)
            candidates.append((score, role, reason))
        candidates.sort(reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else (0.0, "", "")
        confidence = "High" if best[0] >= 0.85 else "Medium" if best[0] >= 0.60 else "Low"
        if best[0] < 0.35:
            best_role = "Unmapped"
            confidence = "Low"
        else:
            best_role = best[1]
        rows.append(
            {
                "CSV column": column,
                "Detected role": best_role,
                "Confidence": confidence,
                "Reason": best[2],
                "Score": best[0],
                "Runner-up": second[1] if second[1] else "—",
                "Runner-up score": second[0],
            }
        )
    return pd.DataFrame(rows)


def default_mapping(inference: pd.DataFrame) -> dict[str, str | None]:
    mapping = {role: None for role in SEMANTIC_ROLES}
    used: set[str] = set()
    for _, row in inference.sort_values("Score", ascending=False).iterrows():
        role = row["Detected role"]
        column = row["CSV column"]
        if role != "Unmapped" and role not in used and column not in used:
            mapping[role] = column
            used.add(role)
    return mapping


# =============================================================================
# 05. PERIOD / CANONICALIZATION / VALIDATION
# =============================================================================


def parse_period(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() >= 0.60:
        return parsed

    text = series.astype("string").str.strip()
    patterns = [
        (r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})$", r"\1 1, \2"),
        (r"^(\d{4})[-/]([01]?\d)$", r"\1-\2-01"),
        (r"^(\d{4})(\d{2})$", r"\1-\2-01"),
    ]
    for pattern, replacement in patterns:
        candidate = pd.to_datetime(text.str.replace(pattern, replacement, regex=True), errors="coerce")
        if candidate.notna().mean() > parsed.notna().mean():
            parsed = candidate
    return parsed


def validate_mapping(raw: pd.DataFrame, mapping: dict[str, str | None]) -> list[Finding]:
    findings: list[Finding] = []
    if raw.empty:
        return [Finding("Blocking", "Empty dataset", "The CSV contains no rows.")]
    if raw.columns.duplicated().any():
        findings.append(Finding("Blocking", "Duplicate column names", "CSV columns must have unique names."))

    mapped = [c for c in mapping.values() if c]
    duplicates = sorted({c for c in mapped if mapped.count(c) > 1})
    if duplicates:
        findings.append(
            Finding(
                "Blocking",
                "One column mapped to multiple roles",
                f"Resolve duplicate mappings: {', '.join(map(str, duplicates))}.",
            )
        )

    for role in ["quantity", "unit_price"]:
        column = mapping.get(role)
        if not column:
            findings.append(
                Finding(
                    "Blocking",
                    f"Missing {human_role(role)}",
                    "The demand model requires positive quantity and unit price.",
                )
            )
            continue
        x = pd.to_numeric(raw[column], errors="coerce")
        if x.notna().mean() < 0.80:
            findings.append(
                Finding("Blocking", f"Too many invalid {human_role(role)} values", f"Column '{column}' has less than 80% valid numeric values.")
            )
        if (x <= 0).any():
            findings.append(
                Finding("Blocking", f"Non-positive {human_role(role)}", f"Column '{column}' contains zero or negative values.")
            )

    unit_price_col = mapping.get("unit_price")
    if unit_price_col:
        p = pd.to_numeric(raw[unit_price_col], errors="coerce")
        unique_prices = p.dropna().nunique()
        if unique_prices < 4:
            findings.append(
                Finding("Blocking", "Insufficient price variation", "At least four distinct positive prices are required for demand-response analysis.")
            )

    if len(raw) < 30:
        findings.append(Finding("Warning", "Small sample", "The dataset is small; prefer pooled or exploratory evidence over fine-grained entity effects."))

    period_col = mapping.get("period")
    if period_col:
        p = parse_period(raw[period_col])
        if p.notna().mean() < 0.60:
            findings.append(
                Finding("Warning", "Weak period parsing", "Less than 60% of period values could be converted into dates; trend/seasonality analysis may be unavailable.")
            )

    stores_col = mapping.get("stores_listed")
    max_stores_col = mapping.get("max_stores")
    if stores_col and max_stores_col:
        stores = pd.to_numeric(raw[stores_col], errors="coerce")
        maximum = pd.to_numeric(raw[max_stores_col], errors="coerce")
        invalid = ((stores < 0) | (maximum <= 0) | (stores > maximum)).sum()
        if invalid:
            findings.append(
                Finding("Warning", "Invalid store coverage", f"{invalid:,} rows have stores listed outside the valid 0–maximum range; distribution will be unavailable for those rows.")
            )

    return findings


@st.cache_data(show_spinner=False)
def canonicalize(raw: pd.DataFrame, mapping_json: str) -> pd.DataFrame:
    mapping = json.loads(mapping_json)
    data = pd.DataFrame({role: raw[column] for role, column in mapping.items() if column})

    numeric_fields = [
        "sales",
        "quantity",
        "unit_price",
        "pack_size",
        "stores_listed",
        "max_stores",
    ]
    for column in numeric_fields:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "period" in data:
        data["period"] = parse_period(data["period"])
        valid_periods = sorted(data["period"].dropna().unique())
        period_index = {p: i for i, p in enumerate(valid_periods)}
        data["time_index"] = data["period"].map(period_index)

    for column in ["retailer", "brand", "package"]:
        if column in data:
            data[column] = data[column].astype("string").fillna("Unknown")

    entity_parts = [c for c in ["brand", "pack_size", "package"] if c in data]
    if entity_parts:
        data["entity_id"] = data[entity_parts].astype("string").fillna("Unknown").agg(" | ".join, axis=1)
    elif "brand" in data:
        data["entity_id"] = data["brand"].astype("string").fillna("Unknown")

    if {"stores_listed", "max_stores"}.issubset(data.columns):
        distribution = data["stores_listed"] / data["max_stores"]
        data["distribution"] = distribution.where((distribution >= 0) & (distribution <= 1))

    if "quantity" in data:
        data["log_quantity"] = np.log(data["quantity"].where(data["quantity"] > 0))
    if "unit_price" in data:
        data["log_unit_price"] = np.log(data["unit_price"].where(data["unit_price"] > 0))
    if "sales" in data:
        total_sales = data["sales"].sum(min_count=1)
        data["sales_contribution"] = data["sales"] / total_sales if total_sales else np.nan
    if "quantity" in data:
        total_quantity = data["quantity"].sum(min_count=1)
        data["quantity_contribution"] = data["quantity"] / total_quantity if total_quantity else np.nan

    return data.reset_index(drop=True)


def capability_report(data: pd.DataFrame) -> dict[str, Capability]:
    def usable(name: str) -> bool:
        return name in data.columns and data[name].notna().any()

    valid_price = (
        usable("quantity")
        and usable("unit_price")
        and data["unit_price"].dropna().nunique() >= 4
        and data["unit_price"].dropna().mean() > 0
        and data["unit_price"].dropna().std() / data["unit_price"].dropna().mean() >= 0.01
    )

    distribution = usable("distribution") and data["distribution"].dropna().nunique() >= 5
    entity_support = (
        "entity_id" in data
        and data["entity_id"].nunique() >= 3
        and data.groupby("entity_id").size().min() >= 8
        and len(data) >= 80
    )
    model_ready = valid_price and len(data.dropna(subset=["quantity", "unit_price"])) >= 30

    return {
        "Performance": Capability(usable("sales") or usable("quantity"), "Requires sales or quantity."),
        "Pricing": Capability(valid_price, "Requires positive quantity, unit price, and meaningful price variation."),
        "Distribution": Capability(distribution, "Requires valid stores-listed and maximum-store coverage."),
        "Entity hierarchy": Capability("entity_id" in data, "Requires a usable entity definition."),
        "Hierarchical price model": Capability(model_ready and entity_support, "Requires at least three supported entities, adequate observations, and within-entity price variation."),
        "Bayesian demand model": Capability(model_ready, "Requires at least 30 valid positive quantity-price observations and price variation."),
        "Scenario analysis": Capability(model_ready, "Requires a fitted Bayesian demand model."),
    }


def model_granularity(data: pd.DataFrame) -> str:
    keys = [c for c in ["entity_id", "retailer", "period"] if c in data.columns and data[c].notna().any()]
    if not keys:
        return "Unidentified analytical grain"
    duplicates = data.duplicated(keys).mean()
    label_map = {"entity_id": "Entity", "retailer": "Retailer", "period": "Period"}
    label_text = " × ".join(label_map[k] for k in keys)
    return label_text + (" (multiple rows per key)" if duplicates > 0.05 else "")


# =============================================================================
# 06. ANALYTICAL SUMMARIES
# =============================================================================


def filter_context(data: pd.DataFrame, key_prefix: str = "ctx") -> pd.DataFrame:
    result = data.copy()
    with st.sidebar:
        st.markdown("### Analysis context")
        for column in ["retailer", "brand", "entity_id"]:
            if column not in result.columns:
                continue
            values = sorted(result[column].dropna().astype(str).unique().tolist())
            if len(values) <= 60:
                selected = st.multiselect(
                    human_role(column),
                    values,
                    default=values,
                    key=f"{key_prefix}_{column}",
                )
                if selected:
                    result = result[result[column].astype(str).isin(selected)]
            else:
                selected = st.text_input(
                    f"Search {human_role(column)}",
                    key=f"{key_prefix}_{column}_search",
                    placeholder="Type to filter...",
                )
                if selected:
                    mask = result[column].astype(str).str.contains(selected, case=False, na=False)
                    result = result[mask]

        if "period" in result.columns and result["period"].notna().any():
            low = result["period"].min().date()
            high = result["period"].max().date()
            selected_range = st.date_input(
                "Period range",
                value=(low, high),
                min_value=low,
                max_value=high,
                key=f"{key_prefix}_period",
            )
            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                result = result[result["period"].between(pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1]))]

    return result


def entity_summary(data: pd.DataFrame) -> pd.DataFrame:
    if "entity_id" not in data.columns:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {}
    if "sales" in data:
        aggregations["sales"] = ("sales", "sum")
    if "quantity" in data:
        aggregations["quantity"] = ("quantity", "sum")
    if "unit_price" in data:
        aggregations["avg_price"] = ("unit_price", "mean")
    if "distribution" in data:
        aggregations["avg_distribution"] = ("distribution", "mean")
    if not aggregations:
        return data[["entity_id"]].drop_duplicates().copy()

    summary = data.groupby("entity_id", dropna=False).agg(**aggregations).reset_index()
    if "sales" in summary:
        total = summary["sales"].sum()
        summary["sales_contribution"] = summary["sales"] / total if total else np.nan
        summary["sales_rank"] = summary["sales"].rank(method="dense", ascending=False).astype("Int64")
    return summary


def entity_growth(data: pd.DataFrame) -> pd.DataFrame:
    if "entity_id" not in data.columns or "period" not in data.columns:
        return pd.DataFrame()

    measures = [c for c in ["sales", "quantity", "unit_price", "distribution"] if c in data.columns]
    if not measures:
        return pd.DataFrame()

    grouped = data.groupby(["entity_id", "period"], dropna=False).agg({c: "sum" if c in {"sales", "quantity"} else "mean" for c in measures}).reset_index()
    grouped = grouped.sort_values(["entity_id", "period"])

    rows = []
    for entity, g in grouped.groupby("entity_id", sort=False):
        if len(g) < 2:
            continue
        current = g.iloc[-1]
        previous = g.iloc[-2]
        row = {"entity_id": entity, "current_period": current["period"], "previous_period": previous["period"]}
        for measure in measures:
            prior = float(previous[measure])
            now = float(current[measure])
            row[f"{measure}_growth"] = (now / prior - 1.0) if np.isfinite(prior) and prior != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_table(data: pd.DataFrame) -> pd.DataFrame:
    base = entity_summary(data)
    growth = entity_growth(data)
    if base.empty:
        return base
    if not growth.empty:
        base = base.merge(growth, on="entity_id", how="left")
        growth_col = "sales_growth" if "sales_growth" in base else "quantity_growth" if "quantity_growth" in base else None
        if growth_col:
            q = base[growth_col].dropna()
            if not q.empty:
                median_growth = q.median()
                median_contribution = base["sales_contribution"].median() if "sales_contribution" in base else np.nan
                conditions = [
                    (base[growth_col] >= median_growth) & (base["sales_contribution"] >= median_contribution),
                    (base[growth_col] < median_growth) & (base["sales_contribution"] >= median_contribution),
                    (base[growth_col] >= median_growth) & (base["sales_contribution"] < median_contribution),
                ]
                base["decision_class"] = np.select(
                    conditions,
                    ["Scale", "Defend", "Develop"],
                    default="Review",
                )
    return base.sort_values("sales", ascending=False) if "sales" in base else base


# =============================================================================
# 07. PYMC MODEL CONSTRUCTION
# =============================================================================


def zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    std = max(std, 1e-8)
    return (values - mean) / std, mean, std


def entity_price_support(data: pd.DataFrame) -> pd.DataFrame:
    if "entity_id" not in data.columns:
        return pd.DataFrame()
    rows = []
    for entity, group in data.groupby("entity_id", dropna=False):
        p = group["unit_price"].dropna()
        rows.append(
            {
                "entity_id": entity,
                "observations": len(group),
                "price_points": p.nunique(),
                "price_mean": p.mean(),
                "price_cv": p.std() / p.mean() if len(p) > 1 and p.mean() else np.nan,
                "price_range": p.max() - p.min() if not p.empty else np.nan,
                "estimable": bool(len(group) >= 8 and p.nunique() >= 4 and (p.std() / p.mean() if len(p) > 1 and p.mean() else 0) >= 0.02),
            }
        )
    return pd.DataFrame(rows)


def choose_model_complexity(data: pd.DataFrame) -> dict[str, bool]:
    supports = entity_price_support(data)
    valid_entities = supports[supports["estimable"]] if not supports.empty else pd.DataFrame()
    hierarchical = len(data) >= 100 and len(valid_entities) >= 3
    retailer_effect = False
    if "retailer" in data.columns:
        counts = data.groupby("retailer").size()
        retailer_effect = len(counts) >= 2 and counts.min() >= 12 and len(data) >= 80
    distribution = "distribution" in data.columns and data["distribution"].dropna().nunique() >= 5
    time = "time_index" in data.columns and data["time_index"].dropna().nunique() >= 8
    season = "period" in data.columns and data["period"].notna().sum() >= 80 and data["period"].dt.month.nunique() >= 6
    return {
        "hierarchical_price": hierarchical,
        "retailer_effect": retailer_effect,
        "distribution": distribution,
        "time": time,
        "season": season,
    }


def fit_bayesian_model(data: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    required = ["quantity", "unit_price"]
    subset = data.dropna(subset=required).copy()
    subset = subset[(subset["quantity"] > 0) & (subset["unit_price"] > 0)]

    if len(subset) < 30:
        raise ValueError("Fewer than 30 valid positive quantity-price observations remain.")

    complexity = choose_model_complexity(subset)

    log_price = np.log(subset["unit_price"].to_numpy(float))
    price_centered = log_price - float(np.mean(log_price))
    price_scale = float(np.std(log_price))
    price_scale = max(price_scale, 1e-8)

    predictor_data: dict[str, np.ndarray] = {"price_centered": price_centered}
    standardization: dict[str, float] = {
        "log_price_mean": float(np.mean(log_price)),
        "log_price_sd": price_scale,
    }

    if complexity["distribution"]:
        distribution = subset["distribution"].fillna(subset["distribution"].median()).to_numpy(float)
        zd, dm, ds = zscore(distribution)
        predictor_data["distribution"] = zd
        standardization.update(distribution_mean=dm, distribution_sd=ds)

    if complexity["time"]:
        time_values = subset["time_index"].to_numpy(float)
        zt, tm, ts = zscore(time_values)
        predictor_data["time"] = zt
        standardization.update(time_mean=tm, time_sd=ts)

    entity_codes = None
    entity_levels: list[str] = []
    if "entity_id" in subset.columns:
        entity_codes, entity_levels = pd.factorize(subset["entity_id"].astype(str))

    retailer_codes = None
    retailer_levels: list[str] = []
    if complexity["retailer_effect"]:
        retailer_codes, retailer_levels = pd.factorize(subset["retailer"].astype(str))

    season_codes = None
    season_levels: list[int] = []
    if complexity["season"]:
        season_codes, season_levels = pd.factorize(subset["period"].dt.month.astype(int))

    coords: dict[str, Any] = {
        "obs": np.arange(len(subset)),
    }
    if entity_codes is not None and complexity["hierarchical_price"]:
        coords["entity"] = entity_levels
    if retailer_codes is not None:
        coords["retailer"] = retailer_levels
    if season_codes is not None:
        coords["season"] = season_levels

    y = subset["quantity"].to_numpy(float)
    log_y = np.log(y)

    with pm.Model(coords=coords, coords_mutable={"obs": np.arange(len(subset))}) as model:
        x_price = pm.Data("x_price", predictor_data["price_centered"], dims="obs")

        alpha = pm.Normal("alpha", mu=float(np.mean(log_y)), sigma=1.5)

        if complexity["hierarchical_price"] and entity_codes is not None:
            entity_idx = pm.Data("entity_idx", entity_codes.astype("int64"), dims="obs")
            mu_elasticity = pm.Normal("mu_elasticity", mu=-1.0, sigma=0.75)
            sigma_elasticity = pm.HalfNormal("sigma_elasticity", sigma=0.50)
            z_elasticity = pm.Normal("z_elasticity", mu=0.0, sigma=1.0, dims="entity")
            elasticity_entity = pm.Deterministic(
                "elasticity_entity",
                mu_elasticity + sigma_elasticity * z_elasticity,
                dims="entity",
            )
            price_term = elasticity_entity[entity_idx] * x_price
        else:
            mu_elasticity = pm.Normal("mu_elasticity", mu=-1.0, sigma=0.75)
            price_term = mu_elasticity * x_price

        mu = alpha + price_term

        if "distribution" in predictor_data:
            x_distribution = pm.Data("x_distribution", predictor_data["distribution"], dims="obs")
            beta_distribution = pm.Normal("beta_distribution", mu=0.0, sigma=0.75)
            mu = mu + beta_distribution * x_distribution

        if "time" in predictor_data:
            x_time = pm.Data("x_time", predictor_data["time"], dims="obs")
            beta_time = pm.Normal("beta_time", mu=0.0, sigma=0.50)
            mu = mu + beta_time * x_time

        if complexity["season"] and season_codes is not None:
            season_idx = pm.Data("season_idx", season_codes.astype("int64"), dims="obs")
            sigma_season = pm.HalfNormal("sigma_season", sigma=0.30)
            z_season = pm.Normal("z_season", mu=0.0, sigma=1.0, dims="season")
            season_effect = pm.Deterministic("season_effect", sigma_season * z_season, dims="season")
            mu = mu + season_effect[season_idx]

        if complexity["retailer_effect"] and retailer_codes is not None:
            retailer_idx = pm.Data("retailer_idx", retailer_codes.astype("int64"), dims="obs")
            sigma_retailer = pm.HalfNormal("sigma_retailer", sigma=0.50)
            z_retailer = pm.Normal("z_retailer", mu=0.0, sigma=1.0, dims="retailer")
            retailer_effect = pm.Deterministic("retailer_effect", sigma_retailer * z_retailer, dims="retailer")
            mu = mu + retailer_effect[retailer_idx]

        sigma = pm.HalfNormal("sigma", sigma=0.60)
        expected_quantity = pm.Deterministic("expected_quantity", pm.math.exp(mu), dims="obs")
        pm.LogNormal("quantity_obs", mu=mu, sigma=sigma, observed=y, dims="obs")

        prior = pm.sample_prior_predictive(
            samples=settings["prior_draws"],
            random_seed=settings["seed"],
        )

        idata = pm.sample(
            draws=settings["draws"],
            tune=settings["tune"],
            chains=settings["chains"],
            cores=1,
            target_accept=settings["target_accept"],
            random_seed=settings["seed"],
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["expected_quantity", "quantity_obs"],
            random_seed=settings["seed"],
            progressbar=False,
            extend_inferencedata=True,
        )

    return {
        "idata": idata,
        "model": model,
        "data": subset,
        "settings": settings,
        "complexity": complexity,
        "standardization": standardization,
        "entities": entity_levels,
        "retailers": retailer_levels,
        "season_levels": season_levels,
        "prior": prior,
        "fingerprint": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# 08. POSTERIOR / DIAGNOSTICS / PREDICTIONS
# =============================================================================


def posterior_flat(idata: Any, variable: str) -> np.ndarray:
    return np.asarray(idata.posterior[variable]).reshape(-1)


def posterior_entity_variable(idata: Any, variable: str) -> np.ndarray:
    arr = np.asarray(idata.posterior[variable])
    if arr.ndim < 3:
        raise ValueError(f"Posterior variable '{variable}' does not contain an entity dimension.")
    return arr.reshape(-1, arr.shape[-1])


def get_elasticity_draws(result: dict[str, Any], entity: str | None = None) -> tuple[np.ndarray, str]:
    idata = result["idata"]
    entities = result["entities"]

    if "elasticity_entity" in idata.posterior and entity is not None and entity in entities:
        draws = posterior_entity_variable(idata, "elasticity_entity")[:, entities.index(entity)]
        return draws, "Entity-level partially pooled posterior"

    draws = posterior_flat(idata, "mu_elasticity")
    return draws, "Population-level pooled posterior"


def sampling_diagnostics(idata: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = az.summary(idata, kind="diagnostics", round_to=3)
    sample_stats = idata.sample_stats

    divergence_count = int(np.asarray(sample_stats["diverging"]).sum()) if "diverging" in sample_stats else 0
    bfmi_values = np.asarray(az.bfmi(idata), dtype=float)
    min_bfmi = float(np.nanmin(bfmi_values)) if bfmi_values.size else np.nan

    rhat_values = summary["r_hat"].dropna() if "r_hat" in summary else pd.Series(dtype=float)
    ess_values = summary["ess_bulk"].dropna() if "ess_bulk" in summary else pd.Series(dtype=float)

    max_rhat = float(rhat_values.max()) if not rhat_values.empty else np.nan
    min_ess = float(ess_values.min()) if not ess_values.empty else np.nan

    if divergence_count > 0 or (np.isfinite(max_rhat) and max_rhat > 1.05):
        status = "Sampling concerns"
    elif (np.isfinite(max_rhat) and max_rhat > 1.01) or (np.isfinite(min_ess) and min_ess < 200) or (np.isfinite(min_bfmi) and min_bfmi < 0.30):
        status = "Minor concerns"
    else:
        status = "Healthy"

    metrics = {
        "status": status,
        "divergences": divergence_count,
        "bfmi": min_bfmi,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
    }
    return summary, metrics


def model_adequacy(result: dict[str, Any]) -> dict[str, Any]:
    idata = result["idata"]
    if "posterior_predictive" not in idata or "quantity_obs" not in idata.posterior_predictive:
        return {"status": "Unavailable", "coverage": np.nan}

    observed = result["data"]["quantity"].to_numpy(float)
    predicted = np.asarray(idata.posterior_predictive["quantity_obs"]).reshape(-1, len(observed))
    median = np.median(predicted, axis=0)
    lower = np.quantile(predicted, 0.05, axis=0)
    upper = np.quantile(predicted, 0.95, axis=0)
    coverage = float(np.mean((observed >= lower) & (observed <= upper)))
    corr = safe_corr(pd.Series(np.log(observed)), pd.Series(np.log(np.clip(median, 1e-12, None))))
    status = "Good" if coverage >= 0.85 and np.isfinite(corr) and corr >= 0.75 else "Mixed" if coverage >= 0.70 else "Weak"
    return {"status": status, "coverage": coverage, "correlation": corr, "observed": observed, "median": median, "lower": lower, "upper": upper}


def posterior_predict_batch(
    result: dict[str, Any],
    row: pd.Series,
    prices: np.ndarray,
    distributions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized posterior prediction for many scenarios in one PyMC call."""
    model = result["model"]
    idata = result["idata"]
    entities = result["entities"]
    retailers = result["retailers"]
    stats = result["standardization"]
    complexity = result["complexity"]

    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n == 0:
        return np.empty((0,)), np.empty((0,))

    updates: dict[str, Any] = {
        "x_price": np.log(prices) - stats["log_price_mean"],
    }

    if "x_distribution" in model.named_vars and distributions is not None:
        dists = np.asarray(distributions, dtype=float)
        if len(dists) != n:
            raise ValueError("Price and distribution scenario arrays must have the same length.")
        updates["x_distribution"] = (dists - stats["distribution_mean"]) / stats["distribution_sd"]

    if complexity["time"] and "x_time" in model.named_vars and "time_index" in row.index and pd.notna(row["time_index"]):
        ztime = (float(row["time_index"]) - stats["time_mean"]) / stats["time_sd"]
        updates["x_time"] = np.full(n, ztime, dtype=float)

    if "entity_idx" in model.named_vars:
        entity = str(row["entity_id"])
        if entity not in entities:
            raise ValueError(f"Entity '{entity}' is not represented in the fitted model.")
        updates["entity_idx"] = np.full(n, entities.index(entity), dtype="int64")

    if "retailer_idx" in model.named_vars:
        retailer = str(row["retailer"])
        if retailer not in retailers:
            raise ValueError(f"Retailer '{retailer}' is not represented in the fitted model.")
        updates["retailer_idx"] = np.full(n, retailers.index(retailer), dtype="int64")

    if "season_idx" in model.named_vars and pd.notna(row.get("period")):
        month = int(pd.Timestamp(row["period"]).month)
        seasons = list(result.get("season_levels", []))
        if month in seasons:
            updates["season_idx"] = np.full(n, seasons.index(month), dtype="int64")

    with model:
        pm.set_data(updates, coords={"obs": np.arange(n)})
        prediction_idata = pm.sample_posterior_predictive(
            idata,
            var_names=["expected_quantity", "quantity_obs"],
            predictions=True,
            random_seed=result["settings"]["seed"],
            progressbar=False,
            return_inferencedata=True,
        )

    group = prediction_idata.predictions
    expected = np.asarray(group["expected_quantity"]).reshape(-1, n)
    realized = np.asarray(group["quantity_obs"]).reshape(-1, n)
    return expected, realized


def posterior_predict(
    result: dict[str, Any],
    row: pd.Series,
    price: float,
    distribution: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    distributions = None if distribution is None else np.array([distribution], dtype=float)
    expected, realized = posterior_predict_batch(result, row, np.array([price], dtype=float), distributions)
    return expected[:, 0], realized[:, 0]


# =============================================================================
# 09. VISUAL HELPERS
# =============================================================================


def status_box(title: str, value: str, detail: str = "", kind: str = "neutral") -> None:
    colors = {
        "positive": ("#1B7F4B", "#EAF6EF"),
        "warning": ("#A66A00", "#FFF4D6"),
        "negative": ("#B42318", "#FEECEB"),
        "neutral": ("#344054", "#F2F4F7"),
    }
    foreground, background = colors.get(kind, colors["neutral"])
    st.markdown(
        f"""
        <div style="padding:0.85rem 1rem;border-radius:0.6rem;background:{background};margin-bottom:0.6rem;">
            <div style="font-size:0.80rem;color:#667085;">{title}</div>
            <div style="font-size:1.15rem;font-weight:650;color:{foreground};">{value}</div>
            <div style="font-size:0.82rem;color:#667085;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def uncertainty_band_figure(x: np.ndarray, median: np.ndarray, lower: np.ndarray, upper: np.ndarray, x_title: str, y_title: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(76,120,168,0.18)",
            line=dict(width=0),
            name="90% credible interval",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=median, mode="lines", name="Posterior median", line=dict(width=3)))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


# =============================================================================
# 10. PAGES
# =============================================================================


def page_overview(data: pd.DataFrame, model_result: dict[str, Any] | None, capabilities: dict[str, Capability]) -> None:
    st.title("Market Modeling")
    st.caption("Bayesian demand intelligence with explicit uncertainty and research diagnostics.")

    metrics = []
    if "sales" in data:
        metrics.append(("Sales", f"${data['sales'].sum():,.0f}"))
    if "quantity" in data:
        metrics.append(("Quantity", f"{data['quantity'].sum():,.0f}"))
    if "unit_price" in data:
        metrics.append(("Average price", f"${data['unit_price'].mean():.2f}"))
    if "distribution" in data:
        metrics.append(("Average distribution", fmt_pct(data["distribution"].mean())))
    metrics.append(("Entities", str(data["entity_id"].nunique() if "entity_id" in data else 0)))

    cols = st.columns(min(5, max(1, len(metrics))))
    for col, (name, value) in zip(cols, metrics):
        col.metric(name, value)

    st.divider()

    left, right = st.columns([1.35, 1])
    with left:
        if "period" in data.columns and "sales" in data.columns:
            trend = data.dropna(subset=["period"]).groupby("period", as_index=False)["sales"].sum()
            st.plotly_chart(
                px.line(trend, x="period", y="sales", markers=True, title="Sales trend", template="plotly_white"),
                use_container_width=True,
            )
        else:
            st.info("A time series is unavailable because a usable period field was not detected.")

    with right:
        if "sales" in data and "entity_id" in data:
            top = entity_summary(data).nlargest(10, "sales")
            st.plotly_chart(
                px.bar(top, x="sales", y="entity_id", orientation="h", title="Largest contributors", template="plotly_white"),
                use_container_width=True,
            )

    st.subheader("Analytical readiness")
    cap_rows = [
        {"Analysis": name, "Status": "Available" if cap.available else "Unavailable", "Reason": cap.reason}
        for name, cap in capabilities.items()
    ]
    st.dataframe(pd.DataFrame(cap_rows), hide_index=True, use_container_width=True)

    if model_result is None:
        status_box(
            "Bayesian model",
            "Not fitted",
            "Use Model Health / the sidebar control to fit the demand model once the data are validated.",
            "neutral",
        )
    else:
        _, sampling = sampling_diagnostics(model_result["idata"])
        adequacy = model_adequacy(model_result)
        kind = "positive" if sampling["status"] == "Healthy" else "warning"
        status_box("Sampling", sampling["status"], f"R-hat max {sampling['max_rhat']:.3f} · divergences {sampling['divergences']}", kind)
        status_box("Predictive adequacy", adequacy["status"], f"90% predictive coverage {fmt_pct(adequacy.get('coverage', np.nan))}", "positive" if adequacy["status"] == "Good" else "warning")

    st.info(
        "Interpretation note: price and distribution effects are model-implied observational relationships unless a stronger causal identification strategy is supplied by the data."
    )


def page_performance(data: pd.DataFrame) -> None:
    st.title("Performance")
    table = portfolio_table(data)
    if table.empty:
        st.warning("No entity-level performance table can be constructed from this dataset.")
        return

    if "decision_class" in table:
        valid = table.dropna(subset=["sales_contribution"])
        growth_col = "sales_growth" if "sales_growth" in valid else "quantity_growth" if "quantity_growth" in valid else None
        if growth_col and len(valid) >= 4:
            fig = px.scatter(
                valid,
                x=growth_col,
                y="sales_contribution",
                size="sales" if "sales" in valid else None,
                color="decision_class",
                hover_name="entity_id",
                title="Portfolio decision matrix",
                labels={growth_col: "Growth", "sales_contribution": "Sales contribution"},
                template="plotly_white",
            )
            fig.update_xaxes(tickformat=".1%")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough sequential history is available for a growth-based decision matrix.")

    search = st.text_input("Search entities", placeholder="Start typing...")
    display = table.copy()
    if search:
        display = display[display["entity_id"].astype(str).str.contains(search, case=False, na=False)]

    st.dataframe(display, hide_index=True, use_container_width=True)
    st.download_button(
        "Download performance CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name="performance.csv",
        mime="text/csv",
    )


def page_pricing(data: pd.DataFrame, result: dict[str, Any]) -> None:
    st.title("Pricing intelligence")

    entities = sorted(result["entities"])
    if not entities:
        st.warning("No modeled entities are available.")
        return

    entity = st.selectbox("Modeled entity", entities)
    subset = data[data["entity_id"].astype(str) == entity].copy()

    if subset.empty:
        st.warning("No observations are available for this entity.")
        return

    if "retailer" in subset.columns and subset["retailer"].nunique() > 1:
        retailer = st.selectbox("Retailer context", sorted(subset["retailer"].dropna().astype(str).unique()))
        subset = subset[subset["retailer"].astype(str) == retailer]

    row = subset.sort_values("period").iloc[-1] if "period" in subset.columns and subset["period"].notna().any() else subset.iloc[-1]
    elasticity_draws, interpretation = get_elasticity_draws(result, entity)
    support = entity_price_support(data)
    support_row = support[support["entity_id"] == entity]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current price", f"${row['unit_price']:.2f}")
    col2.metric("Posterior elasticity", f"{np.median(elasticity_draws):.2f}")
    col3.metric("P(elasticity < 0)", fmt_pct(np.mean(elasticity_draws < 0)))
    col4.metric("P(elasticity < -1)", fmt_pct(np.mean(elasticity_draws < -1)))

    st.caption(
        f"{interpretation} · 90% credible interval [{np.quantile(elasticity_draws, 0.05):.2f}, {np.quantile(elasticity_draws, 0.95):.2f}]"
    )

    if not support_row.empty:
        sr = support_row.iloc[0]
        reliability = "High" if sr["estimable"] and sr["observations"] >= 18 and sr["price_cv"] >= 0.05 else "Medium" if sr["estimable"] else "Insufficient evidence"
        status_box(
            "Entity price evidence",
            reliability,
            f"{int(sr['observations'])} observations · {int(sr['price_points'])} price points · price CV {fmt_pct(sr['price_cv'])}",
            "positive" if reliability == "High" else "warning" if reliability == "Medium" else "negative",
        )
        if reliability == "Insufficient evidence":
            st.warning("The hierarchical model can still use this entity through partial pooling, but entity-specific evidence is limited. Treat its posterior as shrinkage-informed rather than strongly data-dominant.")

    st.subheader("Price-response analysis")
    current_price = float(row["unit_price"])
    low_price = max(0.01, current_price * 0.75)
    high_price = current_price * 1.25
    price_grid = np.linspace(low_price, high_price, 25)

    base_dist = float(row["distribution"]) if "distribution" in row.index and pd.notna(row.get("distribution")) else None
    dist_grid = None if base_dist is None else np.full(len(price_grid), base_dist, dtype=float)
    expected_draws, _ = posterior_predict_batch(result, row, price_grid, dist_grid)
    expected_rows = [
        (
            float(price),
            float(np.median(expected_draws[:, idx])),
            float(np.quantile(expected_draws[:, idx], 0.05)),
            float(np.quantile(expected_draws[:, idx], 0.95)),
        )
        for idx, price in enumerate(price_grid)
    ]
    curve = pd.DataFrame(expected_rows, columns=["price", "quantity", "q_lo", "q_hi"])
    st.plotly_chart(
        uncertainty_band_figure(
            curve["price"].to_numpy(),
            curve["quantity"].to_numpy(),
            curve["q_lo"].to_numpy(),
            curve["q_hi"].to_numpy(),
            "Unit price",
            "Expected quantity",
            "Posterior expected demand response",
        ),
        use_container_width=True,
    )

    revenue = curve["quantity"] * curve["price"]
    best_idx = int(np.argmax(revenue))
    best_price = float(curve.iloc[best_idx]["price"])
    comparison_prices = np.array([current_price, best_price], dtype=float)
    comparison_dist = None if base_dist is None else np.array([base_dist, base_dist], dtype=float)
    comparison_expected, _ = posterior_predict_batch(result, row, comparison_prices, comparison_dist)
    baseline_expected = comparison_expected[:, 0]
    scenario_expected = comparison_expected[:, 1]
    baseline_revenue = baseline_expected * current_price
    scenario_revenue = scenario_expected * best_price

    a, b, c = st.columns(3)
    a.metric("Posterior-median revenue peak", f"${best_price:.2f}")
    b.metric("Expected revenue change vs current", fmt_pct(np.median(scenario_revenue / baseline_revenue - 1)))
    c.metric("P(revenue improves)", fmt_pct(np.mean(scenario_revenue > baseline_revenue)))

    st.caption("The revenue peak is a posterior scenario comparison, not a guaranteed optimal price. Uncertainty and model specification matter.")


def page_distribution(data: pd.DataFrame, result: dict[str, Any]) -> None:
    st.title("Distribution intelligence")
    if "distribution" not in result["data"].columns or "beta_distribution" not in result["idata"].posterior:
        st.warning("Distribution response was not included in the fitted model.")
        return

    entities = sorted(result["entities"])
    entity = st.selectbox("Modeled entity", entities, key="distribution_entity")
    subset = data[data["entity_id"].astype(str) == entity].copy()
    if subset.empty:
        return
    row = subset.sort_values("period").iloc[-1] if "period" in subset.columns and subset["period"].notna().any() else subset.iloc[-1]
    current_distribution = float(row["distribution"])

    dist_grid = np.linspace(0.20, 1.0, 25)
    prices = np.full(len(dist_grid), float(row["unit_price"]), dtype=float)
    expected_draws, _ = posterior_predict_batch(result, row, prices, dist_grid)
    curve = pd.DataFrame({
        "distribution": dist_grid,
        "quantity": np.median(expected_draws, axis=0),
        "q_lo": np.quantile(expected_draws, 0.05, axis=0),
        "q_hi": np.quantile(expected_draws, 0.95, axis=0),
    })

    st.plotly_chart(
        uncertainty_band_figure(
            curve["distribution"].to_numpy(),
            curve["quantity"].to_numpy(),
            curve["q_lo"].to_numpy(),
            curve["q_hi"].to_numpy(),
            "Distribution",
            "Expected quantity",
            "Posterior distribution-response relationship",
        ),
        use_container_width=True,
    )

    current_expected, _ = posterior_predict(result, row, float(row["unit_price"]), current_distribution)
    target_distribution = float(np.clip(current_distribution + 0.10, 0, 1))
    scenario_expected, _ = posterior_predict(result, row, float(row["unit_price"]), target_distribution)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current distribution", fmt_pct(current_distribution))
    c2.metric("Illustrative +10pp response", fmt_pct(np.median(scenario_expected / current_expected - 1)))
    c3.metric("P(improvement)", fmt_pct(np.mean(scenario_expected > current_expected)))

    st.info("Distribution results are model-implied observational relationships. They should not be interpreted as causal incremental sales without stronger identification.")


def page_scenarios(data: pd.DataFrame, result: dict[str, Any]) -> None:
    st.title("Scenario studio")
    st.caption("Compare baseline and alternative commercial conditions through the fitted posterior.")

    entity = st.selectbox("Entity", sorted(result["entities"]), key="scenario_entity")
    subset = data[data["entity_id"].astype(str) == entity].copy()
    if subset.empty:
        return

    if "retailer" in subset.columns and subset["retailer"].nunique() > 1:
        retailer = st.selectbox("Retailer context", sorted(subset["retailer"].dropna().astype(str).unique()), key="scenario_retailer")
        subset = subset[subset["retailer"].astype(str) == retailer]

    row = subset.sort_values("period").iloc[-1] if "period" in subset.columns and subset["period"].notna().any() else subset.iloc[-1]
    base_price = float(row["unit_price"])
    base_distribution = float(row["distribution"]) if "distribution" in row.index and pd.notna(row.get("distribution")) else None

    left, right = st.columns(2)
    scenario_price = left.number_input(
        "Scenario price",
        min_value=0.01,
        max_value=max(0.02, base_price * 3),
        value=base_price,
        step=max(0.01, base_price * 0.01),
    )
    scenario_distribution = None
    if base_distribution is not None and result["complexity"]["distribution"]:
        scenario_distribution = right.slider("Scenario distribution", 0.01, 1.0, min(max(base_distribution, 0.01), 1.0), 0.01)

    run = st.button("Run posterior scenario", type="primary", use_container_width=True)
    if not run:
        baseline_expected, _ = posterior_predict(result, row, base_price, base_distribution)
        st.info(f"Baseline context loaded. Current price ${base_price:.2f}" + (f" · current distribution {fmt_pct(base_distribution)}" if base_distribution is not None else ""))
        return

    scenario_prices = np.array([base_price, float(scenario_price)], dtype=float)
    if base_distribution is None and scenario_distribution is None:
        scenario_dists = None
    else:
        scenario_dists = np.array([base_distribution, scenario_distribution], dtype=float)
    expected_matrix, realized_matrix = posterior_predict_batch(result, row, scenario_prices, scenario_dists)
    baseline_expected = expected_matrix[:, 0]
    scenario_expected = expected_matrix[:, 1]
    scenario_realized = realized_matrix[:, 1]

    baseline_revenue = baseline_expected * base_price
    scenario_revenue = scenario_expected * float(scenario_price)
    baseline_pred_revenue = baseline_expected * base_price
    scenario_pred_revenue = scenario_realized * float(scenario_price)

    rows = []
    for metric, base, scen, base_pred, scen_pred in [
        ("Expected quantity", baseline_expected, scenario_expected, baseline_expected, scenario_expected),
        ("Expected sales", baseline_revenue, scenario_revenue, baseline_revenue, scenario_revenue),
    ]:
        delta = scen / base - 1
        rows.append(
            {
                "Metric": metric,
                "Baseline median": np.median(base),
                "Scenario median": np.median(scen),
                "Median change": np.median(delta),
                "90% interval": f"[{np.quantile(delta, 0.05):.1%}, {np.quantile(delta, 0.95):.1%}]",
                "P(improvement)": np.mean(scen > base),
            }
        )

    display = pd.DataFrame(rows)
    st.dataframe(
        display.style.format(
            {
                "Baseline median": "{:,.0f}",
                "Scenario median": "{:,.0f}",
                "Median change": "{:.1%}",
                "P(improvement)": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Price change", fmt_pct(scenario_price / base_price - 1))
    if base_distribution is not None and scenario_distribution is not None:
        c2.metric("Distribution change", fmt_pct(scenario_distribution - base_distribution))
    else:
        c2.metric("Distribution change", "—")
    c3.metric("P(sales improve)", fmt_pct(np.mean(scenario_revenue > baseline_revenue)))

    st.caption("Expected outcomes use posterior parameter uncertainty. Future realized demand additionally contains observation noise. The model is not refit for a scenario.")


def page_model_health(result: dict[str, Any], synthetic_truth: dict[str, float] | None) -> None:
    st.title("Model Health")
    summary, diagnostics = sampling_diagnostics(result["idata"])
    adequacy = model_adequacy(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampling", diagnostics["status"])
    c2.metric("Max R-hat", f"{diagnostics['max_rhat']:.3f}")
    c3.metric("Min bulk ESS", fmt_number(diagnostics["min_ess"], 0))
    c4.metric("Divergences", str(diagnostics["divergences"]))

    c5, c6 = st.columns(2)
    c5.metric("Minimum BFMI", f"{diagnostics['bfmi']:.2f}")
    c6.metric("Predictive adequacy", adequacy["status"])

    if diagnostics["divergences"] > 0:
        st.error("Divergences were detected. Treat posterior-based decisions as unreliable until the model geometry is addressed.")

    if adequacy["status"] != "Unavailable":
        st.subheader("Posterior predictive check")
        ppc = adequacy
        scatter = pd.DataFrame({
            "Observed": np.log(ppc["observed"]),
            "Predicted median": np.log(np.clip(ppc["median"], 1e-12, None)),
            "Lower": np.log(np.clip(ppc["lower"], 1e-12, None)),
            "Upper": np.log(np.clip(ppc["upper"], 1e-12, None)),
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scatter["Observed"], y=scatter["Predicted median"], mode="markers", name="Observation"))
        mn = min(scatter["Observed"].min(), scatter["Predicted median"].min())
        mx = max(scatter["Observed"].max(), scatter["Predicted median"].max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect fit"))
        fig.update_layout(title="Observed vs posterior predictive median (log scale)", xaxis_title="Observed log quantity", yaxis_title="Predicted log quantity", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"90% posterior predictive coverage: {fmt_pct(ppc['coverage'])} · observed/predicted correlation: {ppc['correlation']:.2f}")

    st.subheader("Model specification")
    complexity = result["complexity"]
    specification = pd.DataFrame(
        [
            ["Response", "Quantity", "LogNormal likelihood on original scale"],
            ["Price", "log(price) centered", "Elasticity parameterization"],
            ["Entity elasticity", "Enabled" if complexity["hierarchical_price"] else "Pooled", "Partial pooling when data support it"],
            ["Retailer effect", "Enabled" if complexity["retailer_effect"] else "Not used", "Hierarchical intercept"],
            ["Distribution", "Enabled" if complexity["distribution"] else "Not used", "Standardized observational predictor"],
            ["Time trend", "Enabled" if complexity["time"] else "Not used", "Standardized time index"],
            ["Seasonality", "Enabled" if complexity["season"] else "Not used", "Monthly hierarchical effects"],
        ],
        columns=["Component", "Status", "Method"],
    )
    st.dataframe(specification, hide_index=True, use_container_width=True)

    st.subheader("Diagnostics by parameter")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Prior predictive calibration")
    prior_values = np.asarray(result["prior"].prior_predictive["quantity_obs"]).reshape(-1)
    observed = result["data"]["quantity"].to_numpy(float)
    st.write(
        f"Prior predictive 5th–95th percentile: {fmt_number(np.quantile(prior_values, 0.05))}–{fmt_number(np.quantile(prior_values, 0.95))} · observed range: {fmt_number(np.min(observed))}–{fmt_number(np.max(observed))}."
    )

    if synthetic_truth:
        page_research_validation(result, synthetic_truth)


def page_research_validation(result: dict[str, Any], truth: dict[str, float]) -> None:
    st.subheader("Research validation — synthetic ground truth")
    if "elasticity_entity" not in result["idata"].posterior:
        st.info("Entity-level elasticity was not enabled for this fit, so ground-truth recovery cannot be assessed at entity level.")
        return

    rows = []
    posterior = posterior_entity_variable(result["idata"], "elasticity_entity")
    for entity, true_value in truth.items():
        if entity not in result["entities"]:
            continue
        idx = result["entities"].index(entity)
        draws = posterior[:, idx]
        lo, med, hi = np.quantile(draws, [0.05, 0.50, 0.95])
        rows.append(
            {
                "Entity": entity,
                "True elasticity": true_value,
                "Posterior median": med,
                "90% interval": f"[{lo:.2f}, {hi:.2f}]",
                "Absolute error": abs(med - true_value),
                "Covered": lo <= true_value <= hi,
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
    if not table.empty:
        st.metric("90% interval coverage", fmt_pct(table["Covered"].mean()))
        st.caption("This research view is only available for the built-in synthetic dataset because the true generating parameters are known there.")


def page_data(
    raw: pd.DataFrame,
    data: pd.DataFrame,
    mapping: dict[str, str | None],
    findings: list[Finding],
    capabilities: dict[str, Capability],
) -> None:
    st.title("Data & methodology")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Columns", f"{len(raw.columns):,}")
    c3.metric("Analytical grain", model_granularity(data))
    c4.metric("Entities", str(data["entity_id"].nunique() if "entity_id" in data else 0))

    st.subheader("Schema mapping")
    mapping_table = pd.DataFrame(
        {
            "Semantic field": list(mapping.keys()),
            "CSV column": [mapping[k] or "—" for k in mapping],
        }
    )
    st.dataframe(mapping_table, hide_index=True, use_container_width=True)

    st.subheader("Data quality")
    finding_table = pd.DataFrame([asdict(f) for f in findings]) if findings else pd.DataFrame([{"severity": "Information", "issue": "No findings", "detail": "No blocking or warning conditions were detected during setup."}])
    st.dataframe(finding_table, hide_index=True, use_container_width=True)

    st.subheader("Missingness")
    missing = data.isna().mean().mul(100).round(1).rename("Missing %").to_frame()
    st.dataframe(missing, use_container_width=True)

    if "sales" in data.columns and "quantity" in data.columns and "unit_price" in data.columns:
        valid = (data["quantity"] > 0) & (data["unit_price"] > 0) & data["sales"].notna()
        ratio = data.loc[valid, "sales"] / (data.loc[valid, "quantity"] * data.loc[valid, "unit_price"])
        st.subheader("Sales consistency")
        st.write(
            f"Median Sales ÷ (Quantity × Unit price): {ratio.median():.3f} · within ±5%: {np.mean(np.abs(ratio - 1) <= 0.05):.1%} · within ±10%: {np.mean(np.abs(ratio - 1) <= 0.10):.1%}"
        )

    st.subheader("Available capabilities")
    cap_table = pd.DataFrame(
        [
            {"Analysis": name, "Available": c.available, "Reason": c.reason}
            for name, c in capabilities.items()
        ]
    )
    st.dataframe(cap_table, hide_index=True, use_container_width=True)

    st.download_button(
        "Download cleaned dataset",
        data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_market_data.csv",
        mime="text/csv",
    )


# =============================================================================
# 11. SETUP / STATE / MAIN UI
# =============================================================================


def initialize_state() -> None:
    defaults = {
        "stage": "choose",
        "raw": None,
        "data": None,
        "mapping": None,
        "findings": [],
        "fingerprint": None,
        "model_result": None,
        "synthetic_truth": None,
        "error_detail": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_dataset() -> None:
    for key, value in {
        "stage": "choose",
        "raw": None,
        "data": None,
        "mapping": None,
        "findings": [],
        "fingerprint": None,
        "model_result": None,
        "synthetic_truth": None,
        "error_detail": None,
    }.items():
        st.session_state[key] = value


def page_choose_data() -> None:
    st.title("Market Modeling")
    st.markdown("### Build a Bayesian demand model from your own CSV")
    st.caption("Upload market data or use the synthetic research sample. The sample follows the same pipeline as user data and includes known ground truth for model-recovery testing.")

    left, right = st.columns(2)
    with left:
        if st.button("Use synthetic research sample", type="primary", use_container_width=True):
            raw, truth = sample_dataset()
            st.session_state.raw = raw
            st.session_state.synthetic_truth = truth
            st.session_state.stage = "mapping"
            st.rerun()
        st.info("The synthetic sample contains repeated entity-retailer-month observations, price variation, distribution variation, seasonality, and known true elasticities.")

    with right:
        upload = st.file_uploader("Upload CSV", type=["csv"], label_visibility="visible")
        if upload is not None:
            try:
                st.session_state.raw = load_csv(upload.getvalue())
                st.session_state.synthetic_truth = None
                st.session_state.stage = "mapping"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")


def page_mapping() -> None:
    raw = st.session_state.raw
    inference = infer_schema(raw)
    defaults = default_mapping(inference)

    st.title("Dataset configuration")
    st.caption("Confirm the semantic meaning of your columns before analytical processing begins.")
    st.write(f"**Dataset:** {len(raw):,} rows × {len(raw.columns):,} columns")

    display_inference = inference.drop(columns=["Score", "Runner-up score"], errors="ignore")
    st.dataframe(display_inference, hide_index=True, use_container_width=True)

    st.subheader("Confirm or override mappings")
    options = ["— Unmapped —"] + list(raw.columns)
    columns = st.columns(2)
    mapping: dict[str, str | None] = {}

    for idx, role in enumerate(SEMANTIC_ROLES):
        default_column = defaults.get(role)
        default_index = options.index(default_column) if default_column in options else 0
        choice = columns[idx % 2].selectbox(
            human_role(role),
            options,
            index=default_index,
            key=f"mapping_{role}",
            help=f"Map a CSV field to the semantic role '{human_role(role)}'.",
        )
        mapping[role] = None if choice == options[0] else choice

    mapped = [c for c in mapping.values() if c]
    duplicates = sorted({c for c in mapped if mapped.count(c) > 1})
    if duplicates:
        st.error(f"One CSV column cannot map to multiple roles: {', '.join(duplicates)}")

    findings = validate_mapping(raw, mapping)
    finding_df = pd.DataFrame([asdict(f) for f in findings]) if findings else pd.DataFrame([{"severity": "Information", "issue": "Validation passed", "detail": "The mapped schema passed initial checks."}])
    st.subheader("Validation")
    st.dataframe(finding_df, hide_index=True, use_container_width=True)

    blocking = any(f.severity == "Blocking" for f in findings) or bool(duplicates)
    if st.button("Confirm dataset", type="primary", disabled=blocking, use_container_width=True):
        canonical = canonicalize(raw, json.dumps(mapping, sort_keys=True))
        fp = fingerprint(raw, mapping)
        st.session_state.data = canonical
        st.session_state.mapping = mapping
        st.session_state.findings = findings
        st.session_state.fingerprint = fp
        st.session_state.model_result = None
        st.session_state.stage = "app"
        st.rerun()


def fit_panel(data: pd.DataFrame, fingerprint_value: str) -> None:
    st.sidebar.subheader("Bayesian model")
    settings = {
        "draws": st.sidebar.select_slider("Posterior draws", options=[300, 500, 800, 1200], value=500),
        "tune": st.sidebar.select_slider("Tuning steps", options=[300, 500, 800, 1200], value=500),
        "chains": st.sidebar.selectbox("Chains", [2, 3, 4], index=0),
        "target_accept": st.sidebar.slider("Target acceptance", 0.85, 0.99, 0.92, 0.01),
        "prior_draws": st.sidebar.select_slider("Prior predictive draws", options=[50, 100, 200], value=100),
        "seed": RANDOM_SEED,
    }

    if st.sidebar.button("Fit Bayesian demand model", type="primary", use_container_width=True):
        try:
            st.session_state.error_detail = None
            with st.status("Fitting Bayesian demand model...", expanded=True) as status:
                st.write("Preparing model structure")
                result = fit_bayesian_model(data, settings)
                result["fingerprint"] = fingerprint_value
                st.session_state.model_result = result
                status.update(label="Bayesian model fitted", state="complete", expanded=False)
            st.rerun()
        except Exception as exc:
            st.session_state.error_detail = traceback.format_exc()
            st.error(f"Model fitting failed: {exc}")

    if st.session_state.error_detail:
        with st.expander("Technical error details"):
            st.code(st.session_state.error_detail)


def main_app() -> None:
    data = st.session_state.data
    capabilities = capability_report(data)

    st.sidebar.title("Market Modeling")
    st.sidebar.caption(f"App {APP_VERSION}")
    st.sidebar.button("Change dataset", on_click=reset_dataset, use_container_width=True)

    if capabilities["Bayesian demand model"].available and st.session_state.model_result is None:
        fit_panel(data, st.session_state.fingerprint)
    elif not capabilities["Bayesian demand model"].available:
        st.sidebar.warning(capabilities["Bayesian demand model"].reason)

    analysis_data = filter_context(data)
    model_result = st.session_state.model_result

    pages = ["Overview", "Performance"]
    if model_result is not None and capabilities["Pricing"].available:
        pages.append("Pricing")
    if model_result is not None and capabilities["Distribution"].available:
        pages.append("Distribution")
    if model_result is not None:
        pages.append("Scenarios")
        pages.append("Model Health")
    pages.append("Data")

    selected_page = st.sidebar.radio("Navigate", pages)
    st.sidebar.caption(f"{len(analysis_data):,} observations in current context")

    if selected_page == "Overview":
        page_overview(analysis_data, model_result, capabilities)
    elif selected_page == "Performance":
        page_performance(analysis_data)
    elif selected_page == "Pricing":
        page_pricing(analysis_data, model_result)
    elif selected_page == "Distribution":
        page_distribution(analysis_data, model_result)
    elif selected_page == "Scenarios":
        page_scenarios(analysis_data, model_result)
    elif selected_page == "Model Health":
        page_model_health(model_result, st.session_state.synthetic_truth)
    elif selected_page == "Data":
        page_data(
            st.session_state.raw,
            st.session_state.data,
            st.session_state.mapping,
            st.session_state.findings,
            capabilities,
        )


def main() -> None:
    initialize_state()
    if st.session_state.stage == "choose":
        page_choose_data()
    elif st.session_state.stage == "mapping":
        page_mapping()
    else:
        main_app()


if __name__ == "__main__":
    main()
