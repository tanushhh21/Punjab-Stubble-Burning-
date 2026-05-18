"""
Execute ONLY Phases 0, 1, 2 of modeling_master.ipynb.
Uses nbconvert in-process with a cell-index cutoff so we never touch Phase 3+.
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NB_PATH = 'modeling_master.ipynb'

with open(NB_PATH) as f:
    nb = nbformat.read(f, as_version=4)

# Identify which cells belong to Phases 0/1/2 vs 3+
# Phase 3 starts at the markdown "## Phase 3"
# Strategy: collect cells up to (but not including) the Phase 3 markdown
EXEC_CELLS = []
SKIP_CELLS = []
in_exec    = True

for cell in nb.cells:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    # Stop executing as soon as we see Phase 3 header
    if cell['cell_type'] == 'markdown' and '## Phase 3' in src:
        in_exec = False
    if in_exec:
        EXEC_CELLS.append(cell)
    else:
        SKIP_CELLS.append(cell)

print(f"Cells to execute  (Phases 0-2): {len(EXEC_CELLS)}")
print(f"Cells to skip     (Phases 3-10): {len(SKIP_CELLS)}")

# Build a mini-notebook of only the exec cells
exec_nb = nbformat.v4.new_notebook()
exec_nb.cells = EXEC_CELLS
exec_nb.metadata = nb.metadata

# Execute
ep = ExecutePreprocessor(timeout=120, kernel_name='python3')
print("\nExecuting Phases 0, 1, 2...")
ep.preprocess(exec_nb, {'metadata': {'path': os.getcwd()}})
print("Execution complete.")

# Merge executed output back into full notebook
for i, cell in enumerate(nb.cells):
    if i < len(EXEC_CELLS):
        nb.cells[i] = exec_nb.cells[i]
    # Phase 3+ cells stay untouched (no outputs, execution_count=None)

# Save
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)

print(f"\nSaved: {NB_PATH}")

# ── Report outputs ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PHASE 0/1/2 EXECUTION REPORT")
print("=" * 60)

for cell in exec_nb.cells:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if cell['cell_type'] != 'code':
        continue
    if not cell['outputs']:
        continue
    print(f"\n--- Cell (first 60 chars: {src[:60].strip()!r}) ---")
    for out in cell['outputs']:
        if out.get('output_type') in ('stream', 'execute_result', 'display_data'):
            text = out.get('text', '') or ''.join(out.get('data', {}).get('text/plain', ''))
            if isinstance(text, list):
                text = ''.join(text)
            print(text.strip())
