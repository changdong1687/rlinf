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

"""Offline tests for BAC: DP schedule, BU, schedule IO, and the block-cache
mechanism (residual reuse + cond/uncond separation) via fake numpy modules."""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bac_dreamzero import (  # noqa: E402
    BACPolicy,
    bubbling_union,
    load_schedule,
    optimal_update_steps,
)
from block_cache import install_block_cache  # noqa: E402
from cache_common import CacheStats  # noqa: E402


# ----------------------------- ACS DP ----------------------------- #
def test_dp_step0_always_included():
    S = np.eye(4) + 0.01
    steps = optimal_update_steps(S, num_caches=2, metric="cosine")
    assert 0 in steps
    assert len(steps) <= 2


def test_dp_identical_steps_need_one_cache():
    # all steps identical => one anchor (step 0) suffices; DP fills toward num_caches
    S = np.ones((5, 5))
    steps = optimal_update_steps(S, num_caches=1, metric="cosine")
    assert steps == [0]


def test_dp_block_diagonal_picks_segment_boundary():
    # two clusters {0,1} and {2,3}: within-cluster sim 1, across 0
    S = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=float,
    )
    steps = optimal_update_steps(S, num_caches=2, metric="cosine")
    assert steps == [0, 2], steps


def test_dp_num_caches_clamped():
    S = np.ones((3, 3))
    steps = optimal_update_steps(S, num_caches=99, metric="cosine")
    assert max(steps) < 3 and 0 in steps


# ----------------------------- BU ----------------------------- #
def test_bubbling_union_pulls_deeper_steps_up():
    sched = {0: [0], 1: [0, 5], 2: [0, 8]}
    out = bubbling_union(sched, bu_blocks=[0], num_blocks=3)
    # block 0 should absorb steps from blocks >= 0 (itself, 1, 2)
    assert out[0] == [0, 5, 8]
    # untouched blocks unchanged
    assert out[1] == [0, 5] and out[2] == [0, 8]


def test_bubbling_union_no_blocks_is_noop():
    sched = {0: [0], 1: [0, 3]}
    out = bubbling_union(sched, bu_blocks=[], num_blocks=2)
    assert out == {0: [0], 1: [0, 3]}


# ----------------------------- schedule IO ----------------------------- #
def test_load_schedule_coerces_and_forces_step0():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "schedule.json")
        with open(p, "w") as f:
            json.dump(
                {"num_steps": 16, "num_blocks": 2, "num_caches": 3,
                 "schedule": {"0": [3, 7], "1": [2]}},
                f,
            )
        spec = load_schedule(p)
        assert spec["schedule"][0] == [0, 3, 7]  # step 0 force-added
        assert spec["schedule"][1] == [0, 2]


# ----------------------------- BACPolicy ----------------------------- #
def test_bac_policy_lookup():
    pol = BACPolicy({0: [0], 1: [0, 1]})
    assert pol.should_compute(0, 0, branch=0) is True
    assert pol.should_compute(0, 1, branch=0) is False  # block0 reuses at step1
    assert pol.should_compute(1, 1, branch=0) is True
    # unscheduled block always computes
    assert pol.should_compute(5, 3, branch=0) is True


# ---------------- block-cache mechanism with fakes ---------------- #
class FakeBlock:
    def __init__(self, i):
        self.i = i

    def forward(self, x=None, kv_cache=None, **kw):
        return x + float(self.i + 1), f"kv{self.i}"

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class FakeDiT:
    def __init__(self, n):
        self.blocks = [FakeBlock(i) for i in range(n)]

    def forward(self, x):
        for b in self.blocks:
            x, _ = b(x=x, kv_cache=None)
        return x

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class FakeActionHead:
    def __init__(self, dit, num_steps):
        self.model = dit
        self.num_inference_steps = num_steps

    def should_run_model(self, index, current_timestep, prev_predictions):
        return True


class FakeModel:
    def __init__(self, n_blocks, num_steps):
        self.action_head = FakeActionHead(FakeDiT(n_blocks), num_steps)
        self.num_steps = num_steps

    def predict_action_batch(self, x0, mode="eval"):
        outs = []
        for step in range(self.num_steps):
            run = self.action_head.should_run_model(step, step, [])
            if not run:
                continue
            for _branch in range(2):  # cond, uncond
                outs.append(self.action_head.model(np.array([float(x0)])))
        return outs


def test_block_cache_reuse_matches_full_compute_and_counts():
    model = FakeModel(n_blocks=2, num_steps=2)
    stats = CacheStats(method="bac", num_steps=2, num_blocks=2)
    # block0 updates only at step0; block1 updates at both steps.
    install_block_cache(model, BACPolicy({0: [0], 1: [0, 1]}), stats=stats)

    outs = model.predict_action_batch(0.0)  # 2 steps * 2 branches = 4 outputs
    # full compute of x=0 through 2 blocks = 0 + 1 + 2 = 3 every time (deterministic),
    # so reuse must reproduce the same value.
    assert all(float(o[0]) == 3.0 for o in outs), [float(o[0]) for o in outs]

    s = stats.summary()
    # step0: 2 blocks * 2 branches computed = 4
    # step1: block0 reused (2 branches), block1 computed (2 branches) = 2 computes
    assert s["mean_block_computes"] == 6.0, s
    assert s["mean_block_total"] == 8.0, s


def test_block_cache_separates_cond_and_uncond():
    model = FakeModel(n_blocks=2, num_steps=2)
    install_block_cache(model, BACPolicy({0: [0], 1: [0, 1]}))
    model.predict_action_batch(0.0)
    ctrl = model._block_cache
    # both CFG branches got their own cached residuals for each block.
    assert set(ctrl.cache.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_block_cache_resets_between_chunks():
    model = FakeModel(n_blocks=2, num_steps=2)
    install_block_cache(model, BACPolicy({0: [0], 1: [0, 1]}))
    model.predict_action_batch(0.0)
    model.predict_action_batch(0.0)  # second chunk must reset, not crash/contaminate
    ctrl = model._block_cache
    assert ctrl.step == 1  # last step of the 2-step loop


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
