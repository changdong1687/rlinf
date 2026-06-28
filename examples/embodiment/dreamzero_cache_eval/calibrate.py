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

"""Automatic calibration / parameter sweep for the DreamZero cache baselines.

For one method it:
  1. (optional) runs the no-cache baseline once -> reference success rate + latency;
  2. sweeps a parameter over a list of values; for EACH value it
       - launches that method's server as a subprocess (own port + --stats-out),
       - waits until the port is open (model loaded),
       - runs the existing LIBERO client at a small scale (few tasks/episodes),
       - stops the server,
       - reads results.json (success) + server_stats.json (latency / skip);
  3. picks the recommended config: **success-first** -> among points whose success
     rate is within ``--tolerance`` of the baseline, the one with the largest
     speedup (baseline_latency / point_latency);
  4. prints a table and writes ``calibrate_results.{json,csv}``.

Pure stdlib (subprocess/socket/json), no torch. **Must run on the GPU machine**
(it drives the real server + LIBERO client). Export DREAMZERO_PATH / LIBERO_ROOT
(and any model env) before running, exactly as for the run_*.sh scripts.

Examples
--------
TeaCache threshold sweep (baseline auto-run):
  DREAMZERO_PATH=... LIBERO_ROOT=... \
  python calibrate.py --method teacache \
    --sweep teacache-thresh=0.08,0.12,0.16,0.22,0.3 \
    --server-args "--model-path $CKPT --metadata-json-path $META --tokenizer-path $TOK --device cuda:0" \
    --client-args "--benchmark-name libero_spatial --task-ids 0 1 2 --n-eval 5" \
    --run-baseline --tolerance 0.02 --out-dir ./runs/calib_teacache

BWCache threshold sweep:
  python calibrate.py --method bwcache --sweep bw-thresh=0.02,0.05,0.08,0.12 ...

BAC over pre-generated schedules (make them first with run_bac_schedule.sh):
  python calibrate.py --method bac \
    --sweep bac-schedule=./runs/sched_nc4.json,./runs/sched_nc6.json,./runs/sched_nc8.json ...
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CLIENT = str(HERE.parent / "dreamzero_libero_eval" / "run_client.sh")

_METHOD_SERVER = {
    "teacache": "run_teacache.sh",
    "bac": "run_bac.sh",
    "bwcache": "run_bwcache.sh",
    "baseline": "run_baseline.sh",
}


# --------------------------------------------------------------------------- #
# subprocess helpers
# --------------------------------------------------------------------------- #
def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, wait_secs: int) -> bool:
    """Poll until the server binds the port (= model loaded) or it dies / times out."""
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # server exited before binding
        if _port_open(host, port):
            return True
        time.sleep(2.0)
    return False


def _start_server(cmd: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    # start_new_session so we can kill the whole process group (run_*.sh -> python).
    return subprocess.Popen(
        shlex.split(cmd),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


# --------------------------------------------------------------------------- #
# one (server, client) measurement
# --------------------------------------------------------------------------- #
def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_point(
    label: str,
    server_cmd: str,
    client_cmd: str,
    out_dir: Path,
    host: str,
    port: int,
    wait_secs: int,
) -> dict:
    """Start a server, run the client, stop the server, collect metrics."""
    point_dir = out_dir / label
    point_dir.mkdir(parents=True, exist_ok=True)
    stats_path = point_dir / "server_stats.json"

    full_server = f"{server_cmd} --port {port} --stats-out {stats_path}"
    full_client = f"{client_cmd} --host {host} --port {port} --output-dir {point_dir}"

    print(f"\n=== [{label}] starting server: {full_server}")
    proc = _start_server(full_server, point_dir / "server.log")
    try:
        if not _wait_for_port(host, port, proc, wait_secs):
            print(f"[{label}] SERVER FAILED TO START (see {point_dir/'server.log'})")
            return {"label": label, "ok": False, "error": "server_start"}
        print(f"[{label}] server ready on {host}:{port}; running client...")
        with open(point_dir / "client.log", "w", encoding="utf-8") as clog:
            rc = subprocess.run(
                shlex.split(full_client), stdout=clog, stderr=subprocess.STDOUT
            ).returncode
        if rc != 0:
            print(f"[{label}] CLIENT FAILED rc={rc} (see {point_dir/'client.log'})")
            return {"label": label, "ok": False, "error": "client_run"}
    finally:
        _stop_server(proc)
        time.sleep(2.0)  # let the port free up before the next point

    results = _read_json(point_dir / "results.json")
    stats = _read_json(stats_path)
    rec = {
        "label": label,
        "ok": results.get("mean_success_rate") is not None,
        "success_rate": results.get("mean_success_rate"),
        "mean_wall_s": stats.get("mean_wall_s"),
        "step_skip_ratio": stats.get("step_skip_ratio"),
        "block_skip_ratio": stats.get("block_skip_ratio"),
        "n_infers": stats.get("n_infers"),
    }
    print(f"[{label}] success={rec['success_rate']} latency={rec['mean_wall_s']}")
    return rec


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def parse_sweep(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise SystemExit(f"--sweep must be NAME=v1,v2,...  got {spec!r}")
    name, values = spec.split("=", 1)
    vals = [v.strip() for v in values.split(",") if v.strip()]
    if not vals:
        raise SystemExit("--sweep needs at least one value")
    return name.strip(), vals


def compute_speedups(points: list[dict], baseline: dict | None) -> None:
    """Annotate each point with speedup = baseline_latency / point_latency (in place)."""
    base_lat = baseline.get("mean_wall_s") if baseline else None
    for p in points:
        if base_lat and p.get("mean_wall_s"):
            p["speedup"] = base_lat / p["mean_wall_s"]
        else:
            p["speedup"] = None


def recommend(
    points: list[dict],
    baseline: dict | None,
    objective: str = "success-first",
    tolerance: float = 0.02,
    min_speedup: float = 2.0,
) -> dict | None:
    """Pick the recommended sweep point (pure; assumes speedups already computed).

    - success-first: among points within ``tolerance`` of the baseline success
      rate, take the largest speedup; fall back to best success if none qualify.
    - speedup-first: among points with speedup >= ``min_speedup``, take the best
      success; fall back to fastest if none qualify.
    - list-only: no recommendation.
    """
    sweep_pts = [
        p for p in points
        if p.get("ok") and p["label"] != "baseline" and p.get("success_rate") is not None
    ]
    if objective == "list-only" or not sweep_pts:
        return None
    if objective == "success-first":
        if baseline and baseline.get("success_rate") is not None:
            floor = baseline["success_rate"] - tolerance
            eligible = [p for p in sweep_pts if p["success_rate"] >= floor and p.get("speedup")]
            if eligible:
                return max(eligible, key=lambda p: p["speedup"])
        return max(sweep_pts, key=lambda p: p["success_rate"])  # fallback / no baseline
    if objective == "speedup-first":
        eligible = [p for p in sweep_pts if (p.get("speedup") or 0) >= min_speedup]
        if eligible:
            return max(eligible, key=lambda p: p["success_rate"])
        return max(sweep_pts, key=lambda p: (p.get("speedup") or 0))
    return None


def resolve_server_base(method: str, run_server: str | None, server_args: str) -> str:
    if run_server:
        base = run_server
    else:
        script = _METHOD_SERVER.get(method)
        if script is None:
            raise SystemExit(f"unknown method {method!r}")
        base = f"bash {HERE / script}"
    return f"{base} {server_args}".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["teacache", "bac", "bwcache"])
    ap.add_argument("--sweep", required=True, help="PARAM=v1,v2,...  (e.g. teacache-thresh=0.1,0.2)")
    ap.add_argument("--server-args", default="", help="Fixed server args: model-path / metadata / tokenizer / device ...")
    ap.add_argument("--client-args", default="--benchmark-name libero_spatial --task-ids 0 --n-eval 3",
                    help="Fixed client args (small scale for calibration).")
    ap.add_argument("--run-server", default=None, help="Override server launcher base (default: bash run_<method>.sh).")
    ap.add_argument("--run-client", default=DEFAULT_CLIENT, help="Client launcher (default: ../dreamzero_libero_eval/run_client.sh).")
    ap.add_argument("--run-baseline", action="store_true", help="Also run the no-cache baseline once for the speedup reference.")
    ap.add_argument("--baseline-args", default=None, help="Baseline server args (default: same as --server-args).")
    ap.add_argument("--baseline-dir", default=None, help="Reuse an already-run baseline dir (results.json+server_stats.json) instead of running it.")
    ap.add_argument("--objective", default="success-first", choices=["success-first", "speedup-first", "list-only"])
    ap.add_argument("--tolerance", type=float, default=0.02, help="success-first: max allowed success drop vs baseline (absolute).")
    ap.add_argument("--min-speedup", type=float, default=2.0, help="speedup-first: required speedup; pick best success among those.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--server-wait-secs", type=int, default=1800)
    ap.add_argument("--out-dir", default="./runs/calibrate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name, values = parse_sweep(args.sweep)
    client_cmd = f"{args.run_client} {args.client_args}".strip()

    points: list[dict] = []

    # --- baseline (reference latency + success) ---
    baseline = None
    if args.baseline_dir:
        bdir = Path(args.baseline_dir)
        baseline = {
            "label": "baseline",
            "ok": True,
            "success_rate": _read_json(bdir / "results.json").get("mean_success_rate"),
            "mean_wall_s": _read_json(bdir / "server_stats.json").get("mean_wall_s"),
        }
        print(f"[baseline] reused {bdir}: success={baseline['success_rate']} latency={baseline['mean_wall_s']}")
    elif args.run_baseline:
        base_args = args.baseline_args if args.baseline_args is not None else args.server_args
        base_cmd = f"bash {HERE / 'run_baseline.sh'} {base_args}".strip()
        baseline = run_point("baseline", base_cmd, client_cmd, out_dir, args.host, args.port, args.server_wait_secs)
    if baseline:
        points.append(baseline)

    # --- sweep ---
    server_base = resolve_server_base(args.method, args.run_server, args.server_args)
    for v in values:
        label = f"{args.method}_{name}_{v}".replace("/", "_")
        server_cmd = f"{server_base} --{name} {v}"
        points.append(run_point(label, server_cmd, client_cmd, out_dir, args.host, args.port, args.server_wait_secs))

    # --- speedup vs baseline + recommendation ---
    compute_speedups(points, baseline)
    recommended = recommend(
        points, baseline,
        objective=args.objective, tolerance=args.tolerance, min_speedup=args.min_speedup,
    )

    _print_table(points, baseline, recommended)
    _write_outputs(out_dir, points, baseline, recommended, args)


def _print_table(points, baseline, recommended) -> None:
    header = ["run", "success", "lat(s)", "speedup", "step_skip", "block_skip", "ok"]
    w = [26, 8, 8, 8, 10, 11, 4]
    line = "  ".join(h.ljust(x) for h, x in zip(header, w))
    print("\n" + line)
    print("-" * len(line))

    def f(v, spec=""):
        if v is None:
            return "-"
        return format(v, spec) if spec else str(v)

    for p in points:
        mark = "  <== recommended" if recommended is p else ""
        cells = [
            p["label"],
            f(p.get("success_rate"), ".4f"),
            f(p.get("mean_wall_s"), ".4f"),
            (f(p["speedup"], ".2f") + "x") if p.get("speedup") else "-",
            f(p.get("step_skip_ratio"), ".3f"),
            f(p.get("block_skip_ratio"), ".3f"),
            "y" if p.get("ok") else "FAIL",
        ]
        print("  ".join(c.ljust(x) for c, x in zip(cells, w)) + mark)

    if recommended:
        print(f"\nRecommended: {recommended['label']}  "
              f"(success={recommended.get('success_rate')}, speedup="
              f"{format(recommended['speedup'], '.2f') if recommended.get('speedup') else '-'}x)")
        if baseline and baseline.get("success_rate") is not None and recommended.get("success_rate") is not None:
            print(f"  success drop vs baseline: {baseline['success_rate'] - recommended['success_rate']:+.4f}")


def _write_outputs(out_dir, points, baseline, recommended, args) -> None:
    summary = {
        "method": args.method,
        "sweep": args.sweep,
        "objective": args.objective,
        "tolerance": args.tolerance,
        "baseline": baseline,
        "recommended": recommended,
        "points": points,
    }
    with open(out_dir / "calibrate_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    import csv

    with open(out_dir / "calibrate_results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["label", "ok", "success_rate", "mean_wall_s",
                                           "speedup", "step_skip_ratio", "block_skip_ratio", "n_infers"])
        wr.writeheader()
        for p in points:
            wr.writerow({k: p.get(k) for k in wr.fieldnames})
    print(f"\nWrote {out_dir/'calibrate_results.json'} and .csv")


if __name__ == "__main__":
    main()
