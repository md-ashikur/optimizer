#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np


def impute_csv(infile, outfile=None):
    if outfile is None:
        base, ext = os.path.splitext(infile)
        outfile = base + '.imputed' + ext

    df = pd.read_csv(infile)
    total_cells = df.shape[0] * df.shape[1]
    missing_before = df.isna().sum().sum()

    # Identify numeric-like columns (those that have at least one convertible numeric value)
    numeric_cols = []
    for col in df.columns:
        conv = pd.to_numeric(df[col], errors='coerce')
        if conv.notna().sum() > 0:
            numeric_cols.append(col)
            df[col] = conv

    # Impute numeric columns by linear interpolation then mean
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        # Fill remaining numeric NaNs with column means
        for col in numeric_cols:
            if df[col].isna().any():
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    df[col] = df[col].fillna(mean_val)
                else:
                    df[col] = df[col].fillna(0)

    # For non-numeric (object) columns: forward-fill, back-fill, then mode
    obj_cols = [c for c in df.columns if c not in numeric_cols]
    for col in obj_cols:
        # Treat empty strings as NaN
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        if df[col].isna().any():
            try:
                mode_val = df[col].mode(dropna=True)
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                else:
                    df[col] = df[col].fillna('')
            except Exception:
                df[col] = df[col].fillna('')

    missing_after = df.isna().sum().sum()
    filled = missing_before - missing_after

    df.to_csv(outfile, index=False)

    return {
        'infile': infile,
        'outfile': outfile,
        'rows': df.shape[0],
        'cols': df.shape[1],
        'total_cells': total_cells,
        'missing_before': int(missing_before),
        'missing_after': int(missing_after),
        'filled': int(filled),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python impute_csv.py "path/to/file.csv" [output.csv]')
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) >= 3 else None
    report = impute_csv(infile, outfile)
    print(f"Processed {report['rows']} rows x {report['cols']} cols ({report['total_cells']} cells)")
    print(f"Missing before: {report['missing_before']}")
    print(f"Missing after: {report['missing_after']}")
    print(f"Cells filled: {report['filled']}")
    print(f"Output written: {report['outfile']}")
