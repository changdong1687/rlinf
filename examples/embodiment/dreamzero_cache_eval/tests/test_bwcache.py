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

"""Offline tests for BWCache's adaptive block-reuse policy (numpy only)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bwcache_dreamzero import BWCachePolicy  # noqa: E402


def _compute_with_note(pol, block, step, branch, rel):
    """Simulate the controller: ask, then (if computed) note the measured change."""
    should = pol.should_compute(block, step, branch)
    if should:
        pol.note(block, step, branch, rel)
    return should


def test_warmup_steps_always_compute():
    pol = BWCachePolicy(num_steps=16, thresh=1.0, reuse_interval=99, last_step=1.0, warmup_steps=2)
    assert pol.should_compute(0, 0, 0) is True
    assert pol.should_compute(0, 1, 0) is True


def test_tail_always_computes():
    pol = BWCachePolicy(num_steps=10, thresh=1.0, reuse_interval=99, last_step=0.8, warmup_steps=0)
    # last_step_idx = round(0.8*10) = 8 ; steps >= 8 always compute
    assert pol.should_compute(0, 8, 0) is True
    assert pol.should_compute(0, 9, 0) is True


def test_small_change_allows_reuse():
    pol = BWCachePolicy(num_steps=16, thresh=0.1, reuse_interval=99, last_step=1.0, warmup_steps=1)
    _compute_with_note(pol, 0, 0, 0, rel=None)  # warmup compute, no prev rel
    _compute_with_note(pol, 0, 1, 0, rel=0.01)  # measured small change
    # next step: change was small (<thresh) -> reuse allowed
    assert pol.should_compute(0, 2, 0) is False


def test_large_change_forces_compute():
    pol = BWCachePolicy(num_steps=16, thresh=0.1, reuse_interval=99, last_step=1.0, warmup_steps=1)
    _compute_with_note(pol, 0, 0, 0, rel=None)
    _compute_with_note(pol, 0, 1, 0, rel=0.5)  # big change
    assert pol.should_compute(0, 2, 0) is True


def test_reuse_interval_forces_periodic_recompute():
    pol = BWCachePolicy(num_steps=16, thresh=1.0, reuse_interval=3, last_step=1.0, warmup_steps=1)
    # Step 0 computes (warmup) but has no previous residual -> rel is None, so the
    # next step is still forced to compute; only after TWO computes is a change
    # measured and reuse becomes possible. Reuse length is then capped by interval.
    _compute_with_note(pol, 0, 0, 0, rel=None)   # compute, last_compute=0, no rel
    _compute_with_note(pol, 0, 1, 0, rel=0.01)   # forced compute, now rel measured
    assert pol.should_compute(0, 2, 0) is False  # 2-1 < 3 -> reuse
    assert pol.should_compute(0, 3, 0) is False  # 3-1 < 3 -> reuse
    assert pol.should_compute(0, 4, 0) is True   # 4-1 >= 3 -> forced recompute


def test_branches_tracked_independently():
    pol = BWCachePolicy(num_steps=16, thresh=0.1, reuse_interval=99, last_step=1.0, warmup_steps=1)
    _compute_with_note(pol, 0, 0, branch=0, rel=None)
    _compute_with_note(pol, 0, 1, branch=0, rel=0.01)
    # cond branch (0) can reuse at step2; uncond branch (1) was never computed -> must compute
    assert pol.should_compute(0, 2, branch=0) is False
    assert pol.should_compute(0, 2, branch=1) is True


def test_reset_clears():
    pol = BWCachePolicy(num_steps=16, thresh=0.1, reuse_interval=99, last_step=1.0, warmup_steps=0)
    _compute_with_note(pol, 0, 0, 0, rel=None)
    _compute_with_note(pol, 0, 1, 0, rel=0.01)
    pol.reset()
    # after reset there is no history -> must compute
    assert pol.should_compute(0, 5, 0) is True


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
