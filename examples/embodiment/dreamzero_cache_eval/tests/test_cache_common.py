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

"""Offline tests for CacheStats (pure stdlib, no GPU/groot)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache_common import CacheStats  # noqa: E402


def test_step_level_skip_ratio():
    stats = CacheStats(method="teacache", num_steps=16, num_blocks=30)
    # one inference that computed 8 of 16 steps
    stats.start_infer()
    for _ in range(8):
        stats.mark_step_compute()
    rec = stats.end_infer()
    assert rec["step_computes"] == 8.0
    s = stats.summary()
    assert s["mean_step_computes"] == 8.0
    assert abs(s["step_skip_ratio"] - 0.5) < 1e-9


def test_block_level_skip_ratio():
    stats = CacheStats(method="bac", num_steps=2, num_blocks=2)
    stats.start_infer()
    # 2 steps * 2 blocks * 2 branches = 8 block calls; computed 5 of them
    computes = [True, True, True, True, True, False, False, False]
    for c in computes:
        stats.mark_block(c)
    stats.end_infer()
    s = stats.summary()
    assert s["mean_block_computes"] == 5.0
    assert s["mean_block_total"] == 8.0
    assert abs(s["block_skip_ratio"] - (1 - 5 / 8)) < 1e-9


def test_mean_over_multiple_infers():
    stats = CacheStats(method="x", num_steps=16)
    for n in (4, 8):
        stats.start_infer()
        for _ in range(n):
            stats.mark_step_compute()
        stats.end_infer()
    s = stats.summary()
    assert s["n_infers"] == 2
    assert s["mean_step_computes"] == 6.0


def test_flush_writes_json():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "server_stats.json")
        stats = CacheStats(method="bwcache", out_path=path, num_steps=16)
        stats.start_infer()
        stats.mark_step_compute()
        stats.end_infer()  # triggers flush
        assert os.path.exists(path)
        import json

        with open(path) as f:
            data = json.load(f)
        assert data["method"] == "bwcache" and data["n_infers"] == 1


def test_empty_summary():
    s = CacheStats(method="none").summary()
    assert s["n_infers"] == 0


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
