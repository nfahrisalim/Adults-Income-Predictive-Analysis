"""
Auto-cast integer-like numeric columns in a CSV locally (no Colab).

Behavior:
- Reads input CSV (default: adult.csv in current folder).
- Treats '?' as missing (NaN) but does not drop rows.
- For each column:
  * If it's numeric float and all non-null values are whole numbers, cast to pandas Int64 (nullable integer).
  * If it's object but safely convertible to integers without introducing new invalids (beyond '?'), cast to Int64.
- Creates a one-time backup: <output>.bak if overwriting the same file.
- Writes back to output (default: in-place overwrite of adult.csv).

Usage (optional):
  python convert_csv_autoint.py [input_csv] [output_csv]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def is_integer_like(series: pd.Series) -> bool:
    """Return True if all non-null numeric values are whole numbers."""
    if series.empty:
        return False
    # Only consider non-null
    s = series.dropna()
    if s.empty:
        return True
    # Must be numeric dtype
    if not pd.api.types.is_numeric_dtype(s):
        return False
    # Check if fractional part is ~0
    frac = np.modf(s.astype(float))[0]
    return np.all(np.isclose(frac, 0.0))


def safe_object_to_numeric_int64(col: pd.Series) -> pd.Series | None:
    """If object column is safely integer-like when converting, return Int64 series, else None."""
    if col.dtype.kind != 'O':
        return None
    # Remember original Na mask including '?' placeholders
    orig_na = col.isna() | (col == '?')
    coerced = pd.to_numeric(col.replace('?', np.nan), errors='coerce')
    # If conversion creates values that were not NaN/'?' originally, that is fine; we only care
    # to ensure non-null are integer-like
    if is_integer_like(coerced):
        return coerced.round(0).astype('Int64')
    return None


def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path('adult.csv')
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_path

    if not in_path.exists():
        print(f"Input CSV not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path, skipinitialspace=True, dtype=str)
    # Normalize '?' to NaN for processing; keep original df for non-target cols
    df_proc = df.copy()

    # Track changes
    converted_cols: list[str] = []

    for col in df_proc.columns:
        s = df_proc[col]

        # First, attempt object->numeric if feasible
        new_int = safe_object_to_numeric_int64(s)
        if new_int is not None:
            df_proc[col] = new_int
            converted_cols.append(col)
            continue

        # If already numeric (e.g., read as number in other CSVs), check integer-like
        # To reach here, s is likely str; but handle numeric too
        try:
            s_num = pd.to_numeric(s.replace('?', np.nan), errors='coerce') if s.dtype.kind == 'O' else s
        except Exception:
            s_num = s

        if pd.api.types.is_numeric_dtype(s_num) and is_integer_like(s_num):
            df_proc[col] = s_num.round(0).astype('Int64')
            converted_cols.append(col)

    # Write out: create backup if overwriting
    if out_path == in_path:
        bak = out_path.with_suffix(out_path.suffix + '.bak')
        if not bak.exists():
            df.to_csv(bak, index=False)
            print(f"Backup created: {bak}")

    # Persist: to preserve dtypes like Int64, let pandas infer with dtype preservation
    df_proc.to_csv(out_path, index=False)

    if converted_cols:
        print("Converted to Int64:", ", ".join(converted_cols))
    else:
        print("No columns required conversion.")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
