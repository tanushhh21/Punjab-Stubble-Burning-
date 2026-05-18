# Punjab Stubble Fire Forecasting — Dashboard

**Course:** AI3011 Machine Learning & Pattern Recognition, Plaksha University, Spring 2026
**Team:** Tanush Kalhan · Aditt Singh · Adityapratap Singh Parmar · Arnav Jain

---

## Quick start

```bash
# 1. Install dependencies (one-time)
pip install streamlit pandas numpy plotly xgboost shap pyarrow

# 2. Generate pre-computed data (one-time, ~60 seconds)
python precompute.py

# 3. Launch the dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser. Cold start is under 3 seconds on
a MacBook with M-series silicon.

---

## Prerequisites

The following files must exist before running `precompute.py`:

| File | Description |
|---|---|
| `punjab_features_master_clean.csv` | Master feature table (56,160 rows x 78 cols) |
| `models/xgb_tuned_final.json` | Trained XGBoost-Tweedie model |
| `outputs/final_metrics.csv` | Tuned model evaluation metrics |
| `outputs/ablation_results.csv` | Feature-family ablation ladder |
| `outputs/shap_importance.csv` | Per-feature SHAP importance |
| `outputs/shap_family_share.csv` | SHAP share by feature family |
| `outputs/strat_mae_master.csv` | MAE stratified by fire-count bucket |
| `outputs/lead_time_results.csv` | PR-AUC and MAE at t+1, t+2, t+4 horizons |
| `outputs/per_district_named.csv` | Per-district PR-AUC and MAE |
| `outputs/counterfactual_scenarios.csv` | Policy counterfactual sensitivity |
| `outputs/morans_i.csv` | Moran's I spatial autocorrelation result |
| `outputs/cpcb_station_aggregates.csv` | CPCB station validation aggregates |

Generate outputs by running `modeling_master.ipynb` (Phases 0-8) and
`polish_additions.ipynb` (Additions 1-5).

---

## What `precompute.py` generates

```
dashboard_data/
  predictions.parquet       2.3 MB — 9,360 test-year grid-cell predictions
  districts.parquet          3 KB — 23 Punjab district centroids
  features_used.csv          1 KB — 64 feature names used by the model
  shap_test_sample.parquet 718 KB — SHAP values for 2,000-row test sample
  all_years_light.parquet  287 KB — light historical data for all 6 years
```

---

## Dashboard tabs

### Overview
Landing page. Two-column layout with project narrative, a hero Punjab density
map (average predicted fires across the 2023 test season), four headline metric
cards (PR-AUC, MAE, Spearman, districts exceeding 0.95 PR-AUC), an 80-word
summary of the full pipeline, and quick-jump navigation cards.

### Historical Replay
Interactive 2023 burning-season scrubber. A `select_slider` over ISO weeks
40-48 (Oct-Nov 2023) controls a 5-stat summary banner and two side-by-side
Plotly scattermapbox maps (predicted vs observed). Below the maps: a
sortable top-50 high-risk cell table and a cell inspector showing SHAP
attribution and an auto-generated explainer paragraph.

### Forecast Simulator
Sidebar filters (district multi-select, week range, predicted fire count
range, confidence threshold radio) update four metric cards, a filtered
density map, a weekly predicted-vs-observed time-series chart, a district
ranking bar chart (orange = error > 20%), and a CSV download button.

### Model Explorer
Nested sub-tabs:

- **Performance** — Stratified MAE bar chart and lead-time decay dual-axis chart
- **Feature Importance** — SHAP family bar chart and top-20 individual features bar chart
- **Ablation** — PR-AUC and MAE across the four feature-set rungs with annotation
- **Validation** — Moran's I metric cards, CPCB scatter plot with Pearson r, counterfactual bar chart, per-district table

### Methodology
Full pipeline flow diagram, data sources table, expandable feature engineering
sections (FIRMS, NDVI, Weather, Policy), the leakage audit narrative with a
PR-AUC progression chart, an 8-check audit list, and a benchmark comparison
table vs Mor & Mor (2023).

---

## Design decisions

- **No inference at runtime.** The XGBoost model is never loaded by `app.py`.
  All predictions come from pre-computed parquet files.
- **Completely offline.** No Mapbox token, no CDN, no Google Fonts URL.
  Maps use Plotly's built-in `open-street-map` style.
- **Custom CSS only.** No Streamlit theme files, no third-party components.
  Punjab Burning palette: `--orange #E8512A`, `--olive #C9D87C`,
  `--blue #5BA3D0` on a `--bg-dark #0F1419` background.
- **Graceful degradation.** If any output CSV is missing, the affected panel
  shows a styled "Data unavailable" notice and the rest of the app continues
  to render normally.

---

## Re-running precompute

If you retrain the model or regenerate outputs, simply re-run:

```bash
python precompute.py
```

Then refresh the browser tab. Streamlit caches are invalidated automatically
on the next server restart.
