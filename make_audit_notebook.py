"""
Converts run_audit.py into leakage_audit_clean_v3.ipynb
by splitting on the ══ section headers.
"""
import json, re, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('run_audit.py') as f:
    src = f.read()

# Remove the os.chdir line — notebook's cwd is set by Jupyter
src = src.replace("os.chdir(os.path.dirname(os.path.abspath(__file__)))\n", "")
# Remove matplotlib Agg backend (Jupyter renders inline)
src = src.replace("matplotlib.use('Agg')\n", "matplotlib.use('inline')\n")

# Split points: every line that starts with the ══ pattern
SPLIT_RE = re.compile(r'(?=^# ══)', re.MULTILINE)
chunks = SPLIT_RE.split(src)

# Also add %matplotlib inline at the top of the first chunk
chunks[0] = "%matplotlib inline\n" + chunks[0]

def make_code_cell(source):
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text + "\n"],
    }

cells = []

# Title markdown
cells.append(make_md_cell(
    "# Leakage Audit — Clean v3 Model\n\n"
    "Reconstructs the clean feature table inline (same logic as `final_v3_modeling.ipynb`) "
    "then runs **8 structural checks + 3 diagnostics** to confirm no data leakage remains.\n\n"
    "| Split | Years | n rows |\n"
    "|-------|-------|--------|\n"
    "| Train | 2018-2021 | 37,440 |\n"
    "| Val | 2022 | 9,360 |\n"
    "| Test | 2023 | 9,360 |\n\n"
    "> **Previous finding:** Original model reported PR-AUC 0.997 — traced to NDVI NaN leakage  \n"
    "> **After fix:** Clean XGB-Tweedie PR-AUC = 0.894 on held-out 2023 data"
))

# Section headers for markdown cells
SECTION_TITLES = {
    'Phase 0':  '## Phase 0 — Artifact Summary',
    'CHECK 1':  '## Check 1 — Cross-year Boundary in Lag Features',
    'CHECK 2':  '## Check 2 — Intensity Features Lagged',
    'CHECK 3':  '## Check 3 — Cartesian Join Sanity',
    'CHECK 4':  '## Check 4 — NDVI Baseline from Train Years Only',
    'CHECK 5':  '## Check 5 — Spatial Overlap (Informational)',
    'CHECK 6':  '## Check 6 — Feature Ablation',
    'CHECK 7':  '## Check 7 — Shuffled-Target Control (Nuclear Test)',
    'CHECK 8':  '## Check 8 — grid_id Not in Features',
    'DIAG A':   '## Diagnostic A — Stratified MAE by True Fire Count',
    'DIAG B':   '## Diagnostic B — SHAP Feature Importance',
    'DIAG C':   '## Diagnostic C — Leaky v2 vs Clean v3 Comparison',
    'FINAL':    '## Final Verdict',
}

for chunk in chunks:
    if not chunk.strip():
        continue
    # Try to match a section title
    matched_title = None
    for key, title in SECTION_TITLES.items():
        if key in chunk[:50].upper():
            matched_title = title
            break
    if matched_title:
        cells.append(make_md_cell(matched_title))
    cells.append(make_code_cell(chunk))

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

out = 'leakage_audit_clean_v3.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written {out}  ({len(cells)} cells)")
