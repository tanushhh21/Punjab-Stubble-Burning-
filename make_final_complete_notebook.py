"""
Creates final_complete.ipynb — a single end-to-end notebook combining:
  Part 1 : EDA (from final_v1.ipynb)
  Part 2 : Feature Engineering v2 (from final_v2.ipynb)
  Part 3 : Modeling — Clean v3 (from final_v3_modeling.ipynb)
  Part 4 : Leakage Audit (from leakage_audit_clean_v3.ipynb)
"""
import json, copy, uuid

def load(nb_name):
    with open(nb_name) as f:
        return json.load(f)['cells']

v1    = load('final_v1.ipynb')
v2    = load('final_v2.ipynb')
v3    = load('final_v3_modeling.ipynb')
audit = load('leakage_audit_clean_v3.ipynb')

# ─────────────────────────────────────────────────────────────
# Helper to make fresh cells (new id, clear outputs)
# ─────────────────────────────────────────────────────────────
def md(text):
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }

def code(src_str):
    return {
        "cell_type": "code",
        "id": str(uuid.uuid4())[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src_str.splitlines()],
    }

def clone(cell):
    """Deep-copy a cell from an existing notebook, assign new id, clear outputs."""
    c = copy.deepcopy(cell)
    c["id"] = str(uuid.uuid4())[:8]
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

def src(cell):
    return "".join(cell["source"])

# ─────────────────────────────────────────────────────────────
# Filter helpers
# ─────────────────────────────────────────────────────────────
def is_import_cell(cell):
    s = src(cell)
    return cell["cell_type"] == "code" and (
        s.strip().startswith("import ") or
        s.strip().startswith("from ") or
        "import pandas" in s or
        "import numpy" in s
    )

def starts_with(cell, text):
    return src(cell).strip().startswith(text)

# ─────────────────────────────────────────────────────────────
# UNIFIED IMPORTS
# ─────────────────────────────────────────────────────────────
IMPORTS = """\
%matplotlib inline
import os, json, re, warnings, glob, joblib
warnings.filterwarnings('ignore')
os.makedirs('models',  exist_ok=True)
os.makedirs('figures', exist_ok=True)

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import xgboost as xgb
import shap

from sklearn.pipeline          import Pipeline
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import (average_precision_score, mean_absolute_error,
                                        mean_squared_error, roc_auc_score, precision_score,
                                        brier_score_loss, f1_score, r2_score)
from scipy.stats               import spearmanr

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'
ORANGE  = '#E8512A'
BLUE    = '#2962FF'
GREY    = '#999999'

# ── Grid / Season constants (shared across all parts) ──────
LAT_MIN, LAT_MAX = 29.7, 32.5
LON_MIN, LON_MAX = 74.0, 76.5
GRID_DEG         = 0.07          # ~7 km per cell
SEASON_WEEKS     = list(range(40, 49))   # ISO weeks 40-48 (Oct-Nov)

TRAIN_YEARS = [2018, 2019, 2020, 2021]
VAL_YEAR    = 2022
TEST_YEAR   = 2023
"""

# ─────────────────────────────────────────────────────────────
# Build cell list
# ─────────────────────────────────────────────────────────────
cells = []

# ── Title ────────────────────────────────────────────────────
cells.append(md("""\
# Punjab Stubble-Burning Fire Prediction — Complete Pipeline

**Dataset:** NASA FIRMS — MODIS C61 + VIIRS NOAA-20 C2 · Punjab, India · Oct–Nov · 2018–2023
**Grid:** 7 km × 7 km (0.07° bins) → 1,040 cells × 6 years × 9 weeks = 56,160 rows
**Target:** `fire_count_weighted` — confidence-weighted weekly fire detections per grid cell
**Split:** Train 2018-2021 | Val 2022 | Test 2023 (strict temporal, no shuffle)

---

| Part | What it covers |
|------|----------------|
| **1 — EDA** | Raw fire data, sensors, confidence, geography, seasonal pattern |
| **2 — Feature Engineering** | Confidence weighting, cartesian expansion, lag/neighbor features |
| **3 — Modeling (Clean v3)** | NDVI from raw MOD13Q1, 5 models, SHAP, per-district metrics |
| **4 — Leakage Audit** | 8 structural checks + 3 diagnostics → 8/8 PASS |

> **Key finding:** Original NDVI sourced from fire-only rows → NaN encoded fire=0 (PR-AUC 0.997).
> After rebuilding NDVI from all-cell raw files, honest XGB-Tweedie PR-AUC = **0.894** on held-out 2023.
"""))

# ── Imports ──────────────────────────────────────────────────
cells.append(md("## Setup — Imports & Constants"))
cells.append(code(IMPORTS))

# ═══════════════════════════════════════════════════════════════
# PART 1 — EDA (from v1)
# Skip: title [0], summary [1], "Imports" md [2], imports code [3]
# Keep: data loading [5,6], cleaning [8], geo [10], seasonal [12],
#       EDA-year [14], EDA-4pm [16], EDA-spatial [18], EDA-weekly [20],
#       feature engineering intro [22,23,24], correlation [26], summary [28,29,30]
# ═══════════════════════════════════════════════════════════════
cells.append(md("""\
---
# Part 1 — Exploratory Data Analysis

Raw NASA FIRMS fire detections → Punjab bounding box → Oct–Nov season → EDA.
"""))

# v1 cells to include (0-indexed):
#   Skip: 0,1,2,3 (title/summary/imports md+code)
#   Skip: 22-30 (v1 feature engineering & summary) — Part 2 does this properly
#   Keep: 4-21 (all EDA: loading, cleaning, geography, seasonal, yearly, 4pm, spatial, weekly)
V1_SKIP = set(range(0, 4)) | set(range(22, 31))
for i, c in enumerate(v1):
    if i in V1_SKIP:
        continue
    cells.append(clone(c))

# Transition note
cells.append(md("""\
> **Note:** The feature table above (v1) contains only fire-active rows and a same-week \
neighbor feature (leaky). Part 2 fixes both issues. The EDA above is purely exploratory.
"""))

# ═══════════════════════════════════════════════════════════════
# PART 2 — Feature Engineering v2
# v2 re-loads raw data with confidence weighting (different from v1 load)
# Skip: title [0], "Imports" md [1], imports code [2]
# Skip NDVI cells [19,20] — superseded by Part 3's raw-file rebuild
# Keep: data load [4], conf weight [6], geo filter [8], grid [10],
#        aggregate [12], cartesian [14], lags [16], neighbors [18],
#        correlation [22], export [24]
# ═══════════════════════════════════════════════════════════════
cells.append(md("""\
---
# Part 2 — Feature Engineering v2

Three critical bug-fixes over v1:
1. **Confidence-as-weight** (continuous 0–1) instead of hard 50% filter
2. **Cartesian expansion** — all 1,040 grid cells × all weeks (adds zero-fire rows)
3. **Neighbor lag** — queries week *t-1* instead of same week *t* (eliminates leakage)

> NDVI is **not** attached here — it is rebuilt from raw MOD13Q1 rasters in Part 3.
"""))

# Skip: title[0], imports md[1], imports code[2], NDVI md[19], NDVI code[20]
#        NDVI-corr md[21], NDVI-corr code[22] (references missing columns)
#        export[24] replaced with custom version below
V2_SKIP = {0, 1, 2, 19, 20, 21, 22, 24}
for i, c in enumerate(v2):
    if i in V2_SKIP:
        continue
    cells.append(clone(c))

# ── Inject clean correlation cell (no NDVI) ──────────────────
cells.append(md("--- \n## 11. Correlation Diagnostics (no NDVI yet — added in Part 3)"))
cells.append(code("""\
v2_features_no_ndvi = [
    'fire_count_weighted',
    'fire_count_last_week', 'same_week_last_year', '3yr_avg',
    'neighbor_fires_last_week', 'neighbor_fires_last_year',
    'avg_frp', 'avg_brightness', 'night_fire_pct',
    'week_of_season',
]

corr_all = (grid_week[v2_features_no_ndvi].corr()['fire_count_weighted']
            .drop('fire_count_weighted'))
corr_sorted = corr_all.abs().sort_values(ascending=False)

print('Feature correlations with fire_count_weighted (ALL rows including zero-fire):')
for feat in corr_sorted.index:
    val = corr_all[feat]
    bar = '█' * int(abs(val) * 40)
    sign = '+' if val >= 0 else '−'
    print(f'  {feat:<32} {sign}{abs(val):.3f}  {bar}')

fig, ax = plt.subplots(figsize=(9, 8))
corr_matrix = grid_week[v2_features_no_ndvi].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax,
            square=True, linewidths=0.4, annot_kws={'size': 8})
ax.set_title('v2 Feature Correlation Matrix (NDVI added in Part 3)',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/fig_v2_correlation_matrix.png', bbox_inches='tight')
plt.show()
"""))

# ── Inject clean export cell (no NDVI columns) ───────────────
cells.append(md("--- \n## 12. Export `punjab_feature_table_v2.csv`\n\nSaves the feature table without NDVI — Part 3 rebuilds NDVI from raw MOD13Q1 rasters and merges it in."))
cells.append(code("""\
export_cols_v2 = [
    'grid_id', 'grid_x', 'grid_y', 'year', 'week', 'week_of_season',
    'fire_count_raw', 'fire_count_weighted', 'fire_count_filtered',
    'fire_count_last_week', 'same_week_last_year', '3yr_avg',
    'neighbor_fires_last_week', 'neighbor_fires_last_year',
    'avg_frp', 'avg_brightness', 'night_fire_pct', 'avg_confidence',
]
grid_week[export_cols_v2].to_csv('punjab_feature_table_v2.csv', index=False)
print(f'Saved punjab_feature_table_v2.csv  ({len(grid_week):,} rows, {len(export_cols_v2)} cols)')

# ── Validation ─────────────────────────────────────────────────
n_grids = grid_week['grid_id'].nunique()
n_years = grid_week['year'].nunique()
n_weeks = grid_week['week'].nunique()
zero_frac = (grid_week['fire_count_weighted'] == 0).mean()
r_neigh = abs(grid_week[['fire_count_weighted','neighbor_fires_last_week']].corr().iloc[0,1])

checks = {
    f'Total rows == {n_grids} × {n_years} years × {n_weeks} weeks':
        len(grid_week) == n_grids * n_years * n_weeks,
    'Zero-fire rows present (30-80%)':
        0.30 < zero_frac < 0.80,
    'neighbor_fires_last_week |corr| < 0.70 (leakage fixed)':
        r_neigh < 0.70,
    'No NaN in target':
        grid_week['fire_count_weighted'].isna().sum() == 0,
    'Target is non-negative':
        (grid_week['fire_count_weighted'] >= 0).all(),
}

print()
print('=' * 55)
print(' PUNJAB STUBBLE FIRE v2 — VALIDATION CHECKLIST')
print('=' * 55)
all_ok = True
for name, passed in checks.items():
    icon = '✓' if passed else '✗ FAIL'
    print(f'  {icon}  {name}')
    if not passed: all_ok = False

assert all_ok, 'Validation failed — see above'
print('\\n  All checks passed ✅  → punjab_feature_table_v2.csv ready for Part 3')
"""))

# ═══════════════════════════════════════════════════════════════
# PART 3 — Modeling (Clean v3)
# v3 loads the CSV saved by Part 2 and rebuilds NDVI from raw files.
# Skip: title [0], "Imports" md [1], imports code [2]
# Keep everything from cell 3 onwards
# ═══════════════════════════════════════════════════════════════
cells.append(md("""\
---
# Part 3 — Modeling Pipeline (Clean v3)

Loads `punjab_feature_table_v2.csv` saved above, then:
- Lags intensity columns (avg_frp, avg_brightness, night_fire_pct) → shift(1)
- Rebuilds NDVI + EVI + velocity + anomaly from raw MOD13Q1 rasters
- Trains 5 models with strict temporal split (no shuffle, no future data)
- SHAP explainability + per-district analysis + comparison figures
"""))

# Skip: title[0], "Imports & Setup" md[1], imports code[2]
#        cell[3] = original "0b" markdown (we inject a custom one)
# Replace cell 4 (drop-NDVI) with a safe version (NDVI cols already absent from our v2 CSV)
V3_SKIP = {0, 1, 2, 3, 4}
for i, c in enumerate(v3):
    if i in V3_SKIP:
        continue
    # Inject patched cell 4 BEFORE the original cell 5 (NDVI rebuild)
    if i == 5:
        cells.append(md("--- \n## 0b. Load v2, Lag Intensity Columns, Drop Any Leaky NDVI\n\nLoads the feature table saved in Part 2. NDVI columns are absent here (Part 2 skipped the leaky v1 NDVI merge) — they will be rebuilt from raw MOD13Q1 rasters in the next cell."))
        cells.append(code("""\
ap = average_precision_score   # alias used in v3 NDVI check cell

df = pd.read_csv('punjab_feature_table_v2.csv')
df = df.sort_values(['grid_id','year','week']).reset_index(drop=True)

# Lag current-week intensity features → prevent same-week leakage
for col in ['avg_frp','avg_brightness','night_fire_pct']:
    df[f'{col}_last_week'] = df.groupby(['grid_id','year'])[col].shift(1)

# Drop raw (current-week) intensity cols and any residual NDVI columns
DROP_COLS = ['avg_frp','avg_brightness','night_fire_pct','avg_confidence',
             'NDVI','EVI','NDVI_baseline','NDVI_anomaly',
             'NDVI_velocity_1wk','NDVI_velocity_2wk','NDVI_velocity_4wk','NDVI_acceleration']
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f'Shape after drops: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
"""))
    cells.append(clone(c))

# ═══════════════════════════════════════════════════════════════
# PART 4 — Leakage Audit
# After Part 3, memory has: df, FEATURES, train, val, test,
# y_train, y_val, y_test, xgb_tweedie, ndvi_agg, TRAIN_YEARS etc.
# We add a bridge cell, then include audit check cells (6-29).
# Skip audit: title[0], preamble[1], phase0[2], imports[3],
#             reconstruction[4,5] — already done in Parts 2&3
# ═══════════════════════════════════════════════════════════════
cells.append(md("""\
---
# Part 4 — Leakage Audit (Clean v3)

Runs **8 structural checks + 3 diagnostics** on the model trained in Part 3.
Uses variables already in memory (`df`, `FEATURES`, `train`, `val`, `test`, `xgb_tweedie`).

| Check | What it tests |
|-------|---------------|
| 1 | No positive lag values at season start (week 40) |
| 2 | Only `_last_week` intensity columns in feature list |
| 3 | Lag values match actual previous-week fire counts (10-row spot-check) |
| 4 | NDVI baseline computed from train+val years only |
| 5 | Spatial overlap structure (informational) |
| 6 | Feature ablation — NDVI lift over position < 0.05 |
| 7 | Shuffled-target control PR-AUC ≈ base rate |
| 8 | `grid_id` (string) not in feature list |
"""))

# Bridge cell — rename xgb_tweedie → m_full, init check_results
cells.append(code("""\
# ── Bridge: expose Part 3 artifacts under names expected by audit ──
m_full       = xgb_tweedie            # full 15-feature XGB-Tweedie model
check_results = {}                     # will be populated by checks 1-8
y_test_bin   = (y_test > 0).astype(int)   # binary fire indicator for test set
print(f'Bridge ready. df={df.shape}, FEATURES={len(FEATURES)}, test rows={len(test)}')
"""))

# Include audit cells 6-29 (the actual checks and diagnostics)
AUDIT_START = 6  # first check cell
for i, c in enumerate(audit):
    if i < AUDIT_START:
        continue
    cells.append(clone(c))

# ═══════════════════════════════════════════════════════════════
# SUMMARY & CONCLUSIONS
# ═══════════════════════════════════════════════════════════════
cells.append(md("""\
---
# Summary & Conclusions

## Problem
Predict **weekly stubble-burning fire counts** per 7 km × 7 km grid cell across Punjab, India
for the Oct–Nov harvest season using satellite remote sensing data (NASA FIRMS + MODIS vegetation indices).

---

## Dataset
| Source | Sensor | Resolution | Period |
|--------|--------|-----------|--------|
| NASA FIRMS | MODIS C61 | 1 km | 2018–2023 |
| NASA FIRMS | VIIRS NOAA-20 C2 | 375 m | 2018–2023 |
| LPDAAC | MOD13Q1 NDVI/EVI | 250 m (16-day) | 2018–2023 |

- **56,160 rows** after cartesian grid expansion (1,040 cells × 6 years × 9 weeks)
- **49.8% zero-fire rows** (model must learn non-burning patterns too)
- **Strict temporal split:** Train 2018–2021 | Val 2022 | Test 2023

---

## Three Critical Bugs Fixed (v1 → v2)

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | Hard confidence threshold (≥50) dropped 30% of real fires | Missed low-confidence evasion fires | Replaced with continuous `conf_weight = confidence/100` |
| 2 | Only fire-active rows in dataset (28k rows, no zeros) | Model never saw what non-burning looks like | Cartesian expansion → 56,160 rows with 49.8% zeros |
| 3 | Neighbor feature used same-week fires (corr=0.908) | Target leakage — you don't have current-week neighbor data at inference | Shifted to week t-1 → corr dropped to 0.530 |

---

## Critical Leakage Found & Fixed (v2 → v3)

**Root cause:** NDVI sourced from `punjab_feature_table_with_ndvi.csv` which only had fire-active rows.
After cartesian expansion, all zero-fire rows got `NDVI = NaN` — making `NDVI is not-NaN` a near-perfect
proxy for `fire > 0`.

| Symptom | Value |
|---------|-------|
| NDVI-NaN flag PR-AUC (leaky) | **0.993** |
| XGB-Tweedie PR-AUC (leaky v2) | **0.997** ← falsely high |
| NDVI-NaN flag PR-AUC (fixed) | 0.548 |
| XGB-Tweedie PR-AUC (clean v3) | **0.894** ← honest |

**Fix:** Rebuilt NDVI + EVI + velocity + anomaly from raw MOD13Q1 rasters covering **all** 1,040 grid cells.

---

## Model Results — Test 2023 (Clean v3)

| Model | PR-AUC | ROC-AUC | MAE | Spearman |
|-------|--------|---------|-----|---------|
| Persistence (baseline) | 0.750 | — | — | — |
| Logistic Regression | 0.900 | ~0.94 | — | — |
| Random Forest | 0.889 | — | 2.6 | — |
| **XGBoost-Tweedie** | **0.894** | **0.944** | **2.5** | **0.61** |
| Hurdle (XGB clf × XGB reg) | 0.880 | — | 2.6 | — |

- **Best model:** XGBoost-Tweedie (Tweedie variance power=1.5, handles zero-inflated counts)
- **+19% PR-AUC** over persistence baseline on unseen 2023 data
- All models trained with **no shuffle**, strict temporal boundary

---

## SHAP Feature Importance (XGB-Tweedie)

| Rank | Feature | Mean |SHAP| | % Total |
|------|---------|-------------|---------|
| 1 | `neighbor_fires_last_week` | 0.419 | 20.5% |
| 2 | `fire_count_last_week` | 0.393 | 19.3% |
| 3 | `week_of_season` | 0.374 | 18.3% |
| 4 | `3yr_avg` | 0.364 | 17.8% |
| 5 | `avg_frp_last_week` | 0.201 | 9.9% |
| — | NDVI/EVI family (4 features) | — | **3.1%** |

> NDVI family = 3.1% of total SHAP (was ~99% in leaky version → confirms fix worked)

---

## Leakage Audit — 8/8 PASS ✅

| Check | Result |
|-------|--------|
| 1. Cross-year boundary clean | ✓ PASS |
| 2. Intensity features lagged | ✓ PASS |
| 3. Cartesian join sanity | ✓ PASS |
| 4. NDVI baseline train-only | ✓ PASS |
| 5. Spatial overlap acknowledged | ✓ PASS |
| 6. Ablation behaves sensibly | ✓ PASS |
| 7. Shuffled-target control | ✓ PASS |
| 8. No grid_id in features | ✓ PASS |

**VERDICT: NO LEAKAGE FOUND. Clean v3 PR-AUC ≈ 0.89 is real signal, not artifact.**

---

## Key Takeaways

1. **Satellite data works for fire prediction** — historical fire patterns (lag features) dominate with 57% of total SHAP
2. **The 4 PM evasion shift is real** — farmers shifted burning post-1:30 PM satellite overpass; using VIIRS (375m, afternoon pass) captures these fires MODIS misses
3. **Zero-inflation matters** — ~50% of cells never burn in a given week; Tweedie regression handles this natively without a separate zero model
4. **NDVI adds modest signal** — 3.1% of SHAP after the leakage fix; useful but not dominant
5. **Temporal generalization holds** — 0.894 PR-AUC on 2023 with a model trained on 2018–2021 confirms the seasonal pattern is stable year-over-year
"""))

# ─────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out = 'final_complete.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written {out}  ({len(cells)} cells)")
print("Breakdown:")
print(f"  Intro + imports : 3")
print(f"  Part 1 EDA      : {len(v1) - len(V1_SKIP)}")
print(f"  Part 2 Feat Eng : {len(v2) - len(V2_SKIP)}")
print(f"  Part 3 Modeling : {len(v3) - len(V3_SKIP)}")
print(f"  Part 4 Audit    : {len(audit) - AUDIT_START + 2}")  # +2 for section md + bridge
