"""
precompute.py — one-time data preparation for the Punjab Fire dashboard.
Run: python precompute.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

os.makedirs('dashboard_data', exist_ok=True)

print("Loading master feature table...")
df = pd.read_csv('punjab_features_master_clean.csv')
print(f"  {df.shape}")

# Feature groups
NDVI_COLS = [c for c in df.columns if 'NDVI' in c or c == 'EVI']
BASE_FIRMS_COLS = [c for c in [
    'fire_count_last_week', 'same_week_last_year', '3yr_avg',
    'neighbor_fires_last_week', 'neighbor_fires_last_year',
    'avg_frp_last_week', 'avg_brightness_last_week', 'night_fire_pct_last_week',
    'week_of_season', 'grid_x', 'grid_y',
] if c in df.columns]
WEATHER_COLS = [c for c in df.columns if any(c.startswith(p) for p in [
    'temp_C', 'dewpoint_C', 'soil_temp_C', 'soil_moisture',
    'wind_speed', 'wind_dir', 'pressure_kpa',
    'rel_humidity', 'vpd', 'is_dry', 'dry_streak', 'fire_weather_index',
])]
POLICY_COLS = [c for c in [
    'super_seeder_available', 'ngt_enforcement_level', 'ex_gratia_announced',
    'election_year', 'crm_funds_central_cr', 'crm_funds_cumulative_cr',
    'msp_paddy_common', 'years_since_crm_scheme',
] if c in df.columns]

print("Loading XGBoost model...")
model = xgb.XGBRegressor()
model.load_model('models/xgb_tuned_final.json')
n = model.n_features_in_
print(f"  n_features_in_: {n}")

candidate_sets = {
    len(BASE_FIRMS_COLS):                                          BASE_FIRMS_COLS,
    len(BASE_FIRMS_COLS + NDVI_COLS):                              BASE_FIRMS_COLS + NDVI_COLS,
    len(BASE_FIRMS_COLS + NDVI_COLS + WEATHER_COLS):               BASE_FIRMS_COLS + NDVI_COLS + WEATHER_COLS,
    len(BASE_FIRMS_COLS + NDVI_COLS + WEATHER_COLS + POLICY_COLS): BASE_FIRMS_COLS + NDVI_COLS + WEATHER_COLS + POLICY_COLS,
}
FEATURES = candidate_sets.get(n, BASE_FIRMS_COLS + NDVI_COLS + WEATHER_COLS)
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Using {len(FEATURES)} features")

# Punjab district lookup
PUNJAB_DISTRICTS = {
    'Amritsar':   (31.6340, 74.8723), 'Tarn Taran': (31.4515, 74.9255),
    'Gurdaspur':  (32.0419, 75.4055), 'Pathankot':  (32.2746, 75.6521),
    'Hoshiarpur': (31.5320, 75.9117), 'Jalandhar':  (31.3260, 75.5762),
    'Kapurthala': (31.3800, 75.3850), 'SBS Nagar':  (31.1300, 76.1170),
    'Rupnagar':   (30.9700, 76.5333), 'Mohali':     (30.7046, 76.7179),
    'Fatehgarh':  (30.6428, 76.3974), 'Patiala':    (30.3398, 76.3869),
    'Sangrur':    (30.2458, 75.8421), 'Mansa':      (29.9988, 75.3933),
    'Bathinda':   (30.2110, 74.9455), 'Barnala':    (30.3787, 75.5462),
    'Ludhiana':   (30.9010, 75.8573), 'Moga':       (30.8237, 75.1715),
    'Faridkot':   (30.6755, 74.7546), 'Firozpur':   (30.9170, 74.6133),
    'Fazilka':    (30.4031, 74.0282), 'Muktsar':    (30.4744, 74.5161),
    'Malerkotla': (30.5260, 75.8810),
}

def nearest_district(lat, lon):
    return min(PUNJAB_DISTRICTS, key=lambda d: (lat - PUNJAB_DISTRICTS[d][0])**2 + (lon - PUNJAB_DISTRICTS[d][1])**2)

GRID_DEG = 0.07
LAT_MIN, LON_MIN = 29.7, 74.0

print("Generating test-year predictions (2023)...")
test = df[df['year'] == 2023].copy().reset_index(drop=True)
test['predicted']  = np.maximum(model.predict(test[FEATURES]), 0)
test['residual']   = test['fire_count_weighted'] - test['predicted']
test['lat']        = LAT_MIN + (test['grid_y'] + 0.5) * GRID_DEG
test['lon']        = LON_MIN + (test['grid_x'] + 0.5) * GRID_DEG
test['district']   = test.apply(lambda r: nearest_district(r['lat'], r['lon']), axis=1)
print(f"  Test rows: {len(test)}")

test.to_parquet('dashboard_data/predictions.parquet', index=False)
print("  Saved dashboard_data/predictions.parquet")

# District centroids
pd.DataFrame([
    {'district': d, 'lat': lat, 'lon': lon}
    for d, (lat, lon) in PUNJAB_DISTRICTS.items()
]).to_parquet('dashboard_data/districts.parquet', index=False)
print("  Saved dashboard_data/districts.parquet")

# Feature list
pd.Series(FEATURES).to_csv('dashboard_data/features_used.csv', index=False, header=['feature'])
print("  Saved dashboard_data/features_used.csv")

# SHAP sample
print("Computing SHAP values for 2000-row test sample...")
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(test), size=min(2000, len(test)), replace=False)
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(test[FEATURES].iloc[sample_idx])
shap_df    = pd.DataFrame(shap_vals, columns=FEATURES)
shap_df['original_idx'] = sample_idx
shap_df['grid_id']      = test['grid_id'].iloc[sample_idx].values
shap_df.to_parquet('dashboard_data/shap_test_sample.parquet', index=False)
print("  Saved dashboard_data/shap_test_sample.parquet")

# Also save full df with all years for historical replay (all years, light cols only)
keep_cols = ['grid_id', 'grid_x', 'grid_y', 'year', 'week', 'week_of_season',
             'fire_count_weighted', 'fire_count_last_week', '3yr_avg']
df[keep_cols].to_parquet('dashboard_data/all_years_light.parquet', index=False)
print("  Saved dashboard_data/all_years_light.parquet")

print("\nPre-computation complete. Run: streamlit run app.py")
