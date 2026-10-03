"""Command line interface: python -m t6se_features --pos P.fasta --neg N.fasta --out results"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .constants import ALL_GROUPS, CORE_GROUPS
from .core import compute_features, get_backend
from .io import load_dataset, write_table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Bastion6-style sequence & physicochemical features for T6SE data")
    ap.add_argument("--pos", required=True, help="FASTA with positive examples (T6SE)")
    ap.add_argument("--neg", required=True, help="FASTA with negative examples (non-T6SE)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--groups", nargs="+", default=list(CORE_GROUPS), choices=ALL_GROUPS,
                    help=f"feature groups (default: {' '.join(CORE_GROUPS)})")
    ap.add_argument("--extra-pypredt6", action="store_true",
                    help="also compute the 17 physicochemical groups taken from pypredt6.py")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--format", default="csv", choices=["csv", "tsv"])
    ap.add_argument("--maxlag", type=int, default=30, help="QSO maxlag (Bastion6: 30)")
    ap.add_argument("--weight", type=float, default=0.1, help="QSO weight (Bastion6: 0.1)")
    ap.add_argument("--max-tokens", type=int, default=400_000, help="batch memory knob: batch_size * max_len")
    args = ap.parse_args(argv)

    groups = list(args.groups)
    if args.extra_pypredt6 and "PYPREDT6_PHYSCHEM" not in groups:
        groups.append("PYPREDT6_PHYSCHEM")

    df = load_dataset(args.pos, args.neg)
    _, backend = get_backend(args.device)
    print(f"{len(df)} sequences ({int(df.label.sum())} positive, {int((df.label == 0).sum())} negative); backend: {backend}")

    t0 = time.perf_counter()
    feats = compute_features(df["seq"].tolist(), ids=df["id"].tolist(), groups=groups,
                             device=args.device, maxlag=args.maxlag, weight=args.weight,
                             max_tokens=args.max_tokens)
    print(f"features computed in {time.perf_counter() - t0:.2f} s")

    meta = pd.DataFrame({"label": df["label"].values, "class": df["class"].values,
                         "length": df["seq"].str.len().values}, index=pd.Index(df["id"], name="id"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for g, table in feats.items():
        p = write_table(pd.concat([meta, table], axis=1), out / g, args.format)
        print(f"  {g:<18} {table.shape[1]:>4} features -> {p}")
    p = write_table(pd.concat([meta] + list(feats.values()), axis=1), out / "all_features", args.format)
    print(f"  {'all_features':<18} {sum(t.shape[1] for t in feats.values()):>4} features -> {p}")


if __name__ == "__main__":
    main()
