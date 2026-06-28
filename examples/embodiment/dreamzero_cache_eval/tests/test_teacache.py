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

"""Offline tests for the TeaCache step-skip decision (numpy only)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from teacache_dreamzero import TeaCacheState, _rel_l1  # noqa: E402


def test_rel_l1_basic():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert _rel_l1(a, b) == 0.0
    assert abs(_rel_l1(np.array([2.0]), np.array([1.0])) - 1.0) < 1e-6


def test_first_step_always_computes():
    st = TeaCacheState(threshold=1e9)
    assert st.decide(np.ones(4)) is True  # step 0 forced


def test_small_change_is_skipped():
    st = TeaCacheState(threshold=0.5, ret_steps=1)
    assert st.decide(np.ones(4)) is True  # warm-up compute, sets prev
    # tiny change -> accumulated < threshold -> skip
    assert st.decide(np.ones(4) * 1.001) is False


def test_large_change_forces_compute():
    st = TeaCacheState(threshold=0.05, ret_steps=1)
    st.decide(np.ones(4))  # compute
    # big jump -> rel-L1 ~1.0 >> 0.05 -> compute, and accumulator resets
    assert st.decide(np.ones(4) * 2.0) is True


def test_accumulation_eventually_triggers():
    st = TeaCacheState(threshold=0.25, ret_steps=1)
    st.decide(np.ones(4))  # compute (prev=1)
    # each step grows ~10% -> rel_l1 ~0.1; accumulates 0.1,0.2,0.3...
    vals = [1.1, 1.21, 1.331, 1.4641]
    decisions = [st.decide(np.array([v] * 4)) for v in vals]
    # first couple skipped, then once accumulated >= 0.25 a compute happens
    assert decisions[0] is False
    assert any(d is True for d in decisions), decisions


def test_cutoff_tail_always_computes():
    st = TeaCacheState(threshold=1e9, num_steps=4, ret_steps=1, cutoff_steps=3)
    out = [st.decide(np.ones(4)) for _ in range(4)]
    # step0 forced (ret), steps 1,2 skippable (huge threshold), step3 forced (cutoff)
    assert out[0] is True and out[3] is True
    assert out[1] is False and out[2] is False


def test_reset_clears_state():
    st = TeaCacheState(threshold=0.5)
    st.decide(np.ones(4))
    st.decide(np.ones(4) * 1.001)
    st.reset()
    assert st.cnt == 0 and st.prev_proxy is None
    assert st.decide(np.ones(4)) is True  # fresh first step computes


def test_polynomial_rescale_amplifies():
    # coeffs [10, 0] => rescale(x) = 10*x, so small raw change crosses threshold faster
    st = TeaCacheState(threshold=0.5, ret_steps=1, coefficients=[10.0, 0.0])
    st.decide(np.ones(4))
    # raw rel_l1 ~0.1 -> rescaled ~1.0 >= 0.5 -> compute
    assert st.decide(np.ones(4) * 1.1) is True


def test_ret_steps_validation():
    try:
        TeaCacheState(ret_steps=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for ret_steps < 1")


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
