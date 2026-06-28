#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge client results.json + server stats.json into one comparison table.

For each run directory it reads:
  * ``results.json``      (from libero_eval_client.py) -> mean_success_rate
  * ``server_stats.json`` (from the cache server, --stats-out) -> DiT timing/skip

and prints a table of success rate, mean inference latency, speedup vs. the
baseline run, and the skip ratios. Pure stdlib (no numpy/torch), so it runs
anywhere.

Usage:
  python summarize.py \
    --run baseline=./runs/baseline \
    --run teacache=./runs/teacache \
    --run bac=./runs/bac \
    --run bwcache=./runs/bwcache \
    --baseline baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(label: str, run_dir: str) -> dict:
    d = Path(run_dir)
    results = _load(d / "results.json") or {}
    stats = _load(d / "server_stats.json") or {}
    return {
        "label": label,
        "success_rate": results.get("mean_success_rate"),
        "n_tasks": len(results.get("tasks", [])),
        "method": stats.get("method", results.get("server_metadata", {}).get("cache_method", "?")),
        "mean_wall_s": stats.get("mean_wall_s"),
        "mean_step_computes": stats.get("mean_step_computes"),
        "mean_block_computes": stats.get("mean_block_computes"),
        "step_skip_ratio": stats.get("step_skip_ratio"),
        "block_skip_ratio": stats.get("block_skip_ratio"),
        "n_infers": stats.get("n_infers"),
    }


def _fmt(v, spec="") -> str:
    if v is None:
        return "-"
    if spec:
        return format(v, spec)
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                        help="A run to include, e.g. teacache=./runs/teacache. Repeatable.")
    parser.add_argument("--baseline", type=str, default="baseline",
                        help="Label whose mean_wall_s defines the 1.0x speedup reference.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to also write a CSV.")
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=DIR, got {spec!r}")
        label, run_dir = spec.split("=", 1)
        rows.append(collect(label, run_dir))

    base = next((r for r in rows if r["label"] == args.baseline), None)
    base_wall = base["mean_wall_s"] if base else None

    for r in rows:
        if base_wall and r["mean_wall_s"]:
            r["speedup"] = base_wall / r["mean_wall_s"]
        else:
            r["speedup"] = None

    header = ["run", "method", "success", "lat(s)", "speedup", "step_skip", "block_skip", "n_inf"]
    widths = [12, 9, 8, 8, 8, 10, 11, 6]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [
            r["label"],
            str(r["method"]),
            _fmt(r["success_rate"], ".4f"),
            _fmt(r["mean_wall_s"], ".4f"),
            (_fmt(r["speedup"], ".2f") + "x") if r["speedup"] else "-",
            _fmt(r["step_skip_ratio"], ".3f"),
            _fmt(r["block_skip_ratio"], ".3f"),
            _fmt(r["n_infers"]),
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["label", "method", "success_rate", "mean_wall_s", "speedup",
                            "step_skip_ratio", "block_skip_ratio", "n_infers", "n_tasks"],
            )
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
