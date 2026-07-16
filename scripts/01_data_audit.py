"""
Initial data load and column audit for Bike Share Toronto 2023 ridership data.
Checks schema consistency, null rates, dtypes, and row counts across all 12 monthly files.
"""

import pandas as pd
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_data"
CSV_FILES = sorted(DATA_DIR.glob("Bike share ridership 2023-*.csv"))

# ---------------------------------------------------------------------------
# 1. Per-file header + dtype + null audit (read only 100 rows for speed,
#    then do a full row count via wc-equivalent)
# ---------------------------------------------------------------------------

print(f"Found {len(CSV_FILES)} files\n")

audit_rows = []
all_columns = {}  # file -> list[col]

for path in CSV_FILES:
    month = path.stem.split("2023-")[-1]

    # Full read — Jan–Mar are UTF-8; Apr–Dec have cp1252 en-dashes in station names
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    n_rows = len(df)
    cols = list(df.columns)
    dtypes = df.dtypes.to_dict()
    null_counts = df.isnull().sum().to_dict()
    null_pct = {c: round(null_counts[c] / n_rows * 100, 2) for c in cols}

    all_columns[month] = cols

    audit_rows.append({
        "month": month,
        "n_rows": n_rows,
        "n_cols": len(cols),
        "columns": cols,
        "dtypes": {c: str(v) for c, v in dtypes.items()},
        "null_pct": null_pct,
    })

    print(f"  2023-{month}  |  {n_rows:>7,} rows  |  {len(cols)} cols")

# ---------------------------------------------------------------------------
# 2. Schema consistency check
# ---------------------------------------------------------------------------

print("\n--- Schema consistency across months ---")
reference_cols = audit_rows[0]["columns"]
reference_month = audit_rows[0]["month"]

schema_issues = []
for row in audit_rows[1:]:
    if row["columns"] != reference_cols:
        schema_issues.append(
            f"  2023-{row['month']} differs from 2023-{reference_month}:\n"
            f"    extra:   {set(row['columns']) - set(reference_cols)}\n"
            f"    missing: {set(reference_cols) - set(row['columns'])}"
        )

if schema_issues:
    print("SCHEMA DIFFERENCES FOUND:")
    for issue in schema_issues:
        print(issue)
else:
    print("All 12 files share the same column schema. ✓")

# ---------------------------------------------------------------------------
# 3. Canonical column summary (from January as reference)
# ---------------------------------------------------------------------------

print("\n--- Column audit (January as reference) ---")
ref = audit_rows[0]
print(f"\n{'Column':<35} {'Dtype':<15} {'Null %'}")
print("-" * 60)
for col in ref["columns"]:
    dtype = ref["dtypes"][col]
    null = ref["null_pct"][col]
    flag = " ← HIGH NULL" if null > 5 else ""
    print(f"  {col:<33} {dtype:<15} {null:>5.1f}%{flag}")

# ---------------------------------------------------------------------------
# 4. Aggregate totals
# ---------------------------------------------------------------------------

total_rows = sum(r["n_rows"] for r in audit_rows)
print(f"\n--- Totals ---")
print(f"  Total trips (2023): {total_rows:,}")
print(f"  Peak month:         {max(audit_rows, key=lambda r: r['n_rows'])['month']}  "
      f"({max(r['n_rows'] for r in audit_rows):,} trips)")
print(f"  Quiet month:        {min(audit_rows, key=lambda r: r['n_rows'])['month']}  "
      f"({min(r['n_rows'] for r in audit_rows):,} trips)")

# ---------------------------------------------------------------------------
# 5. Save audit summary to JSON for later reference
# ---------------------------------------------------------------------------

out_path = Path(__file__).parent.parent / "data" / "analysis_data" / "audit_2023.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(audit_rows, f, indent=2)

print(f"\nAudit saved to {out_path}")
