# Market Modeling

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io/)
[![PyMC](https://img.shields.io/badge/PyMC-5.28+-orange.svg)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/yourusername/market-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/market-modeling/actions/workflows/ci.yml)

> **Bayesian demand intelligence with explicit uncertainty and research diagnostics.**

A Streamlit application for building hierarchical Bayesian demand models from market data. Upload your own CSV or use the built-in synthetic research sample with known ground truth for model-recovery testing.

## Live Demo

🌐 **Try it now:** [https://marketmodeling-with-pymc.streamlit.app/](https://marketmodeling-with-pymc.streamlit.app/)

## Features

### 📊 Data Pipeline
- **Schema inference** — Automatic semantic role detection for 10 standard market data fields (retailer, brand, pack size, package, stores listed, max stores, sales, quantity, period, unit price)
- **Flexible column mapping** — Alias-aware matching with confidence scoring
- **Validation & canonicalization** — Comprehensive data quality checks with blocking/warning findings
- **Derived metrics** — Distribution, log-transforms, contributions, time indices

### 🧠 Bayesian Modeling
- **Hierarchical price elasticity** — Partial pooling across entities with population-level shrinkage
- **Multi-component specification** — Retailer effects, distribution response, time trends, monthly seasonality
- **Automatic complexity selection** — Model structure adapts to data richness
- **LogNormal likelihood** — Proper uncertainty on original quantity scale

### 📈 Analysis Pages
| Page | Purpose |
|------|---------|
| **Overview** | KPIs, sales trend, top entities, analytical readiness matrix |
| **Performance** | Portfolio table with growth-based decision matrix (Scale/Defend/Develop/Review) |
| **Pricing** | Entity-level elasticity posteriors, price-response curves, revenue optimization scenarios |
| **Distribution** | Distribution-response relationships with credible intervals |
| **Scenarios** | Compare baseline vs. alternative price/distribution conditions through posterior |
| **Model Health** | Sampling diagnostics (R-hat, ESS, BFMI, divergences), posterior predictive checks, prior calibration |
| **Research Validation** | Ground-truth recovery assessment (synthetic sample only) |
| **Data & Methodology** | Schema mapping, quality findings, missingness, capability matrix |

### 🔬 Research-Grade Diagnostics
- **Sampling health** — R-hat, effective sample size, BFMI, divergence detection
- **Posterior predictive checks** — 90% coverage, observed vs. predicted correlation
- **Prior predictive calibration** — Prior range vs. observed data range
- **Entity-level evidence grading** — High/Medium/Insufficient based on observations, price points, and variation

## Quick Start

### Prerequisites
- Python 3.11+
- uv (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/market-modeling.git
cd market-modeling

# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

### 1. Choose Data Source
- **Synthetic Research Sample** — Pre-built dataset with 7 entities × 3 retailers × 36 months, known true elasticities for validation
- **Upload CSV** — Your own market data (see [Data Format](#data-format) below)

### 2. Confirm Schema Mapping
The app infers semantic roles from column names and data characteristics. Review and override mappings as needed.

### 3. Validate & Fit
- Validation runs automatically — blocking issues must be resolved before fitting
- Adjust MCMC settings in the sidebar (draws, tune, chains, target acceptance)
- Click **Fit Bayesian demand model**

### 4. Explore Results
Navigate pages via the sidebar. All analyses respect the current filter context (retailer, brand, entity, period range).

## Data Format

The app expects a CSV with columns mapping to these semantic roles:

| Role | Description | Example Columns |
|------|-------------|-----------------|
| `retailer` | Retail chain / customer | Retailer, Chain, Customer, Account |
| `brand` | Manufacturer brand | Brand, Manufacturer Brand |
| `pack_size` | Volume / net content | Pack Size, Size, Volume, Net Content |
| `package` | Container / format | Package, Format, Container, Pack Type |
| `stores_listed` | Numeric distribution | Stores Listed, Listed Stores, Distribution Stores |
| `max_stores` | Universe / max stores | Max Stores, Maximum Stores, Total Stores |
| `sales` | Revenue / value sales | Sales, Sales $, Revenue, Net Sales |
| `quantity` | Unit volume | Qty, Quantity, Units, Volume |
| `period` | Time dimension | Date, Month, Week, Period, YearMonth |
| `unit_price` | Price per unit | Price, Unit Price, ASP, Selling Price |

**Minimum requirements:** `quantity` and `unit_price` (both positive, ≥4 distinct prices, ≥30 valid rows).

### Sample Data Structure

```csv
Retailer,Brand,Pack Size,Package,Stores Listed,Max Stores (Retailer),Sales ($),Qty,Month,Price_per_Unit
Walmart,Coca-Cola,2.0,l,4605,4710,1269338.4,450120,Jan 2025,2.82
Target,Pepsi,1.5,l,1810,1950,275502.5,112450,Jan 2025,2.45
...
```

## Architecture

```
app.py                          # Single-file Streamlit application
├── Configuration & Constants   # Semantic roles, aliases, version, seed
├── Data Classes                # Finding, Capability
├── Utilities                   # Normalization, formatting, fingerprinting, correlation
├── Synthetic Data              # Ground-truth sample generator
├── Schema Inference            # Role scoring, inference, default mapping
├── Validation                  # Blocking/warning findings
├── Canonicalization            # Type coercion, period parsing, derived fields
├── Capability Report           # Analysis readiness matrix
├── Analytical Summaries        # Entity summary, growth, portfolio decision matrix
├── Model Construction          # Complexity selection, PyMC model, fitting
├── Posterior & Diagnostics     # Elasticity extraction, sampling diagnostics, PPC
├── Prediction                  # Vectorized posterior prediction for scenarios
├── Visual Helpers              # Status boxes, uncertainty band figures
├── Pages                       # Overview, Performance, Pricing, Distribution, Scenarios, Model Health, Data
└── State & UI                  # Session state, navigation, main entry point
```

## Model Specification

The Bayesian demand model uses a **LogNormal likelihood** on quantity with the following linear predictor on log-scale:

```
log(quantity) ~ α + ε_entity × log(price_centered) + β_dist × distribution + β_time × time + season_month + retailer_effect
```

Where:
- `ε_entity` — Hierarchical elasticity with partial pooling (when ≥3 entities with sufficient price variation)
- `β_dist` — Distribution response coefficient (when distribution data available)
- `β_time` — Linear time trend (when ≥8 time periods)
- `season_month` — Monthly hierarchical seasonality (when ≥80 observations across ≥6 months)
- `retailer_effect` — Retailer-level hierarchical intercept (when ≥2 retailers with ≥12 obs each)

**Priors:**
- `α ~ Normal(mean(log_q), 1.5)`
- `μ_elasticity ~ Normal(-1.0, 0.75)`
- `σ_elasticity ~ HalfNormal(0.50)`
- `β ~ Normal(0.0, 0.75)` (distribution, time)
- `σ_season, σ_retailer ~ HalfNormal(0.30–0.50)`
- `σ ~ HalfNormal(0.60)` (observation noise)

## Development

### Code Quality

```bash
# Format & lint
uv run ruff check --fix .
uv run ruff format .
uv run black .

# Type check
uv run mypy app.py --ignore-missing-imports

# Run tests
uv run pytest -v
```

### Pre-commit Hooks

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Project Structure

```
market-modeling/
├── app.py                 # Main application (single file)
├── pyproject.toml         # Project config, dependencies, tool settings
├── requirements.txt       # Pinned dependencies for deployment
├── sample_data.csv        # Example input data
├── README.md              # This file
├── LICENSE                # MIT license
├── .github/
│   ├── workflows/
│   │   └── ci.yml         # CI/CD pipeline
│   └── dependabot.yml     # Automated dependency updates
├── .pre-commit-config.yaml # Pre-commit hooks
└── tests/
    ├── test_utils.py      # Unit tests for utilities
    ├── test_schema.py     # Schema inference tests
    ├── test_validation.py # Validation tests
    └── test_canonicalize.py # Canonicalization tests
```

## Deployment

### Streamlit Community Cloud
1. Push to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Set `app.py` as entry point
4. Deploy

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py sample_data.csv ./
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501"]
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style (ruff, black, mypy)
- Testing requirements
- Pull request process
- Issue reporting

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this in research, please cite:

```bibtex
@software{market_modeling,
  title = {Market Modeling: Bayesian Demand Intelligence},
  author = {Market Modeling Contributors},
  year = {2024},
  url = {https://github.com/yourusername/market-modeling}
}
```

## Acknowledgments

- [PyMC](https://www.pymc.io/) — Probabilistic programming in Python
- [ArviZ](https://arviz-devs.github.io/arviz/) — Exploratory analysis of Bayesian models
- [Streamlit](https://streamlit.io/) — Rapid data app development
- [Plotly](https://plotly.com/) — Interactive visualizations