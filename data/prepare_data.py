#!/usr/bin/env python3
"""
prepare_data.py
Arrange the already-preprocessed CSVs into the directory layout that the
training scripts expect, i.e. for each dataset:

    dat/<dataset>/proc/df_together.csv     <- *_df_together.csv
    dat/<dataset>/proc/split.csv           <- *_split.csv

The training code (albert_train_src.get_split) reads both with index_col=0,
where the first column is the report `id`. No transformation is applied here;
the files are only copied/renamed into place, so the data is byte-for-byte the
same as what you uploaded.

Usage (run once, from the project root, after uploading the CSVs):
    python data/prepare_data.py --src .        # if the 4 CSVs are in the project root
    python data/prepare_data.py --src /path/to/csvs

Expected source filenames (defaults):
    ailf_df_together.csv, ailf_split.csv, tram_df_together.csv, tram_split.csv
"""
import argparse
import os
import shutil
import sys
import pandas as pd

# (source csv basename for df_together, source csv basename for split, dataset folder name)
DATASETS = [
    ("ailf_df_together.csv", "ailf_split.csv", "Analgesics-induced_acute_liver_failure"),
    ("tram_df_together.csv", "tram_split.csv", "Tramadol-related_mortalities"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".", help="Directory containing the 4 source CSVs (default: current dir)")
    ap.add_argument("--dat", default="dat", help="Destination 'dat' directory (default: ./dat)")
    args = ap.parse_args()

    ok = True
    for df_name, split_name, folder in DATASETS:
        src_df = os.path.join(args.src, df_name)
        src_split = os.path.join(args.src, split_name)
        proc_dir = os.path.join(args.dat, folder, "proc")
        os.makedirs(proc_dir, exist_ok=True)

        for src, dst_name in ((src_df, "df_together.csv"), (src_split, "split.csv")):
            dst = os.path.join(proc_dir, dst_name)
            if not os.path.isfile(src):
                print(f"[MISSING] {src}  -> cannot create {dst}", file=sys.stderr)
                ok = False
                continue
            shutil.copyfile(src, dst)
            # sanity check: confirm it loads with index_col=0 as the training code expects
            n = len(pd.read_csv(dst, index_col=0))
            print(f"[OK] {src}  ->  {dst}   ({n} rows)")

    if not ok:
        print("\nSome files were missing. Place all 4 CSVs in --src and re-run.", file=sys.stderr)
        sys.exit(1)
    print("\nData prepared. Expected layout:")
    for _, _, folder in DATASETS:
        print(f"  {os.path.join(args.dat, folder, 'proc')}/  ->  df_together.csv, split.csv")


if __name__ == "__main__":
    main()
