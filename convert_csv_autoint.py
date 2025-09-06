"""
Auto-cast numeric-like columns in a CSV to integers without listing column names.

Behavior:
- Reads the CSV with whitespace/NA handling (Adult dataset friendly)
- For each column:
  * If values are numeric-like and all non-empty values are integers, cast to nullable Int64
  * If values are numeric-like but not integers, cast to float
  * Otherwise leave as-is (object)
- Overwrites the CSV (creates .bak backup by default)

Usage:
  python convert_csv_autoint.py --csv adult.csv
  python convert_csv_autoint.py --csv adult.csv --no-backup
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def is_numeric_like_series(s: pd.Series) -> tuple[bool, pd.Series]:
    """Return whether series is numeric-like and the numeric-coerced series.

    Treat empty strings and configured NA markers as missing before check.
    """
    # Normalize to string for stripping when object dtype
    s_str = s.astype(str).str.strip() if s.dtype == object else s
    s_num = pd.to_numeric(s_str, errors="coerce")
    # Consider only non-missing values
    non_missing = s_num.notna()
    if non_missing.any() and non_missing.sum() >= (len(s) - s.isna().sum()):
        # All non-missing after coercion are numeric
        return True, s_num
    return False, s_num


def convert_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    changes: dict[str, tuple[str, str]] = {}
    for col in df.columns:
        before = str(df[col].dtype)
        s = df[col]
        is_num_like, s_num = is_numeric_like_series(s)
        if is_num_like:
            # If all numeric values are integer-like (e.g., 1.0), cast to Int64
            if (s_num.dropna() == s_num.dropna().astype(int)).all():
                df[col] = s_num.astype("Int64")
            else:
                df[col] = s_num  # float dtype
        after = str(df[col].dtype)
        if before != after:
            changes[col] = (before, after)
    return df, changes


def process_csv(csv_path: Path, make_backup: bool = True) -> dict[str, tuple[str, str]]:
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    # Read with common Adult dataset options
    df = pd.read_csv(
        csv_path,
        na_values=["", "?", " ?"],
        skipinitialspace=True,
    )

    df, changes = convert_frame(df)

    if make_backup:
        bak = csv_path.with_suffix(csv_path.suffix + ".bak")
        bak.write_bytes(csv_path.read_bytes())
        print(f"Backup written to: {bak}")

    df.to_csv(csv_path, index=False)

    return changes


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Auto-cast numeric-like CSV columns to Int64/float")
    p.add_argument("--csv", default="adult.csv", help="Path to CSV (default: adult.csv)")
    p.add_argument("--no-backup", action="store_true", help="Do not create .bak backup")
    args = p.parse_args(argv)

    changes = process_csv(Path(args.csv), make_backup=not args.no_backup)

    if changes:
        print("Dtype changes:")
        for col, (before, after) in changes.items():
            print(f"- {col}: {before} -> {after}")
    else:
        print("No dtype changes were necessary.")


if __name__ == "__main__":
    main()
