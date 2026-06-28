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

"""Offline tests for calibrate.py selection logic (pure stdlib; no GPU/server)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibrate import compute_speedups, parse_sweep, recommend


def _pts():
    base = {"label": "baseline", "ok": True, "success_rate": 0.96, "mean_wall_s": 0.80}
    pts = [
        base,
        {"label": "teacache_t_0.1", "ok": True, "success_rate": 0.955, "mean_wall_s": 0.60},
        {"label": "teacache_t_0.2", "ok": True, "success_rate": 0.950, "mean_wall_s": 0.40},
        {"label": "teacache_t_0.3", "ok": True, "success_rate": 0.870, "mean_wall_s": 0.30},
        {"label": "teacache_t_0.4", "ok": False, "success_rate": None, "mean_wall_s": None},
    ]
    compute_speedups(pts, base)
    return base, pts


def test_parse_sweep():
    name, vals = parse_sweep("teacache-thresh=0.1, 0.2 ,0.3")
    assert name == "teacache-thresh"
    assert vals == ["0.1", "0.2", "0.3"]


def test_compute_speedups():
    base, pts = _pts()
    assert abs(pts[1]["speedup"] - (0.80 / 0.60)) < 1e-9
    assert pts[4]["speedup"] is None  # failed point


def test_success_first_picks_max_speedup_within_tolerance():
    base, pts = _pts()
    rec = recommend(pts, base, objective="success-first", tolerance=0.02)
    # 0.96-0.02=0.94 floor -> 0.955 and 0.950 qualify; 0.870 excluded.
    # among qualifying, max speedup is t_0.2 (0.80/0.40=2.0x).
    assert rec["label"] == "teacache_t_0.2", rec["label"]


def test_success_first_tight_tolerance():
    base, pts = _pts()
    rec = recommend(pts, base, objective="success-first", tolerance=0.006)
    # floor 0.954 -> only t_0.1 (0.955) qualifies.
    assert rec["label"] == "teacache_t_0.1"


def test_success_first_fallback_best_success_when_none_qualify():
    base, pts = _pts()
    rec = recommend(pts, base, objective="success-first", tolerance=0.0001)
    # nothing within 0.0001 -> fallback to best success among sweep pts (t_0.1=0.955)
    assert rec["label"] == "teacache_t_0.1"


def test_speedup_first():
    base, pts = _pts()
    rec = recommend(pts, base, objective="speedup-first", min_speedup=2.0)
    # speedup >= 2.0: t_0.2 (2.0x) and t_0.3 (2.67x); best success among them = t_0.2 (0.95)
    assert rec["label"] == "teacache_t_0.2"


def test_list_only_returns_none():
    base, pts = _pts()
    assert recommend(pts, base, objective="list-only") is None


def test_no_baseline_success_first_uses_best_success():
    base, pts = _pts()
    compute_speedups(pts, None)  # no baseline -> no speedups
    rec = recommend(pts, None, objective="success-first")
    assert rec["label"] == "teacache_t_0.1"  # best success


def test_failed_points_excluded():
    base, pts = _pts()
    rec = recommend(pts, base, objective="success-first", tolerance=1.0)
    assert rec["ok"] is True and rec["success_rate"] is not None


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
