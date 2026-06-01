#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path


RETURNS_TARGET = "ret_future_1"
VOL_TARGETS = ["rv_future_7", "gk_future_7", "rs_future_7"]
BLOCKS = ["price_only", "price_onchain", "full"]
SEEDS = [111, 222, 333, 444, 555, 666]
NEURAL_MODELS = ["lstm", "gru"]


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Create SLURM array grids for the BTC interval benchmark.")
    p.add_argument("--out-dir", default="configs")
    p.add_argument("--vol-targets", default=",".join(VOL_TARGETS))
    p.add_argument("--blocks", default=",".join(BLOCKS))
    p.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    p.add_argument("--neural-models", default=",".join(NEURAL_MODELS))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    vol_targets = parse_csv_list(args.vol_targets)
    blocks = parse_csv_list(args.blocks)
    seeds = parse_int_list(args.seeds)
    neural_models = parse_csv_list(args.neural_models)

    returns = [
        {"target": RETURNS_TARGET, "block": "price_only", "seed": seed}
        for seed in seeds
    ]
    classical = [
        {"target": target, "block": block, "seed": seed}
        for target, block, seed in product(vol_targets, blocks, seeds)
    ]
    neural = [
        {"target": target, "block": block, "model": model, "seed": seed}
        for target, block, model, seed in product(vol_targets, blocks, neural_models, seeds)
    ]

    paths = {
        "returns": out_dir / "grid_returns_baseline.csv",
        "classical": out_dir / "grid_classical.csv",
        "neural": out_dir / "grid_neural.csv",
    }
    write_rows(paths["returns"], ["target", "block", "seed"], returns)
    write_rows(paths["classical"], ["target", "block", "seed"], classical)
    write_rows(paths["neural"], ["target", "block", "model", "seed"], neural)

    for name, rows in [("returns", returns), ("classical", classical), ("neural", neural)]:
        last = len(rows) - 1
        array = f"0-{last}" if rows else "empty"
        print(f"Wrote {paths[name]} with {len(rows)} tasks; array: {array}")


if __name__ == "__main__":
    main()
