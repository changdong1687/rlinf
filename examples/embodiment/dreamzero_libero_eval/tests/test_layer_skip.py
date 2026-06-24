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

"""Offline unit tests for the DiT layer-skip helpers (pure Python, no torch needed)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from layer_skip import (  # noqa: E402
    apply_layer_skip,
    make_identity_block_forward,
    make_video_skip_block_forward,
    parse_layer_indices,
)


class FakeBlock:
    """Stand-in for a DiT block: forward(x, kv_cache) -> (transformed_x, new_cache)."""

    def __init__(self, idx: int):
        self.idx = idx
        self.calls = 0

    def forward(self, x=None, kv_cache=None, **kwargs):
        self.calls += 1
        return x + 100 + self.idx, f"cache_{self.idx}_updated"

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)


class FakeDiT:
    def __init__(self, n):
        self.blocks = [FakeBlock(i) for i in range(n)]


class FakeActionHead:
    def __init__(self, n):
        self.model = FakeDiT(n)


class FakeModel:
    def __init__(self, n):
        self.action_head = FakeActionHead(n)


def test_parse_single_and_ranges():
    assert parse_layer_indices("3,7,11", 30) == [3, 7, 11]
    assert parse_layer_indices("10-13", 30) == [10, 11, 12, 13]
    assert parse_layer_indices("0, 2-3 , 2", 30) == [0, 2, 3]  # dedup + whitespace
    assert parse_layer_indices("", 30) == []


def test_parse_out_of_range_raises():
    for bad in ("30", "-1", "5,40"):
        try:
            parse_layer_indices(bad, 30)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for spec {bad!r}")


def test_identity_forward_passthrough():
    fwd = make_identity_block_forward()
    x, cache = fwd(x=42, kv_cache="orig")
    assert x == 42 and cache == "orig"
    # positional x fallback
    x2, cache2 = fwd(7)
    assert x2 == 7 and cache2 is None


def test_apply_skips_only_selected_blocks():
    model = FakeModel(30)
    skipped, num_layers = apply_layer_skip(model, "3,7,11")
    assert num_layers == 30
    assert skipped == [3, 7, 11]

    blocks = model.action_head.model.blocks
    # Skipped blocks now pass through; non-skipped transform as before.
    for i, b in enumerate(blocks):
        out_x, out_cache = b(x=0, kv_cache=f"in_{i}")
        if i in skipped:
            assert out_x == 0 and out_cache == f"in_{i}", (i, out_x, out_cache)
        else:
            assert out_x == 100 + i and out_cache == f"cache_{i}_updated", (i, out_x)


def test_apply_empty_spec_is_noop():
    model = FakeModel(30)
    skipped, num_layers = apply_layer_skip(model, None)
    assert skipped == [] and num_layers == 30
    # all blocks still active
    out_x, _ = model.action_head.model.blocks[5](x=0, kv_cache="c")
    assert out_x == 105


def test_video_skip_freezes_video_keeps_action():
    # Block input x = [B, Lq, d] laid out as [video | action], last n_act = action.
    n_act, n_vid, d = 3, 5, 4
    lq = n_vid + n_act
    x_in = np.arange(lq * d, dtype=np.float32).reshape(1, lq, d)

    def orig(x=None, kv_cache=None, **kw):
        return x + 1000.0, "cache"  # every token "updated" by +1000

    wrapped = make_video_skip_block_forward(orig, n_act)
    x_out, cache = wrapped(x=x_in.copy(), kv_cache="in")
    # video rows frozen to input; action rows keep the +1000 update.
    assert np.array_equal(x_out[:, :n_vid], x_in[:, :n_vid]), "video tokens must be frozen"
    assert np.array_equal(x_out[:, n_vid:], x_in[:, n_vid:] + 1000.0), "action tokens must update"
    assert cache == "cache"


def test_apply_video_mode_wraps_only_selected():
    # FakeBlock returns numpy so the video-skip wrapper's shape ops work.
    class VBlock:
        def __init__(self, i):
            self.i = i

        def forward(self, x=None, kv_cache=None, **kw):
            return x + 1.0, "c"

        def __call__(self, *a, **k):
            return self.forward(*a, **k)

    class M:
        class action_head:  # noqa: N801
            action_horizon = 2

            class model:  # noqa: N801
                blocks = [VBlock(i) for i in range(6)]

    m = M()
    skipped, n = apply_layer_skip(m, "1,3", mode="video")
    assert skipped == [1, 3] and n == 6
    x = np.ones((1, 5, 3), dtype=np.float32)  # Lq=5, n_act=2 -> n_vid=3
    # skipped layer: video frozen (==1), action updated (==2)
    out, _ = m.action_head.model.blocks[1](x=x.copy())
    assert np.array_equal(out[:, :3], np.ones((1, 3, 3)))
    assert np.array_equal(out[:, 3:], np.full((1, 2, 3), 2.0))
    # non-skipped layer: everything updated (==2)
    out2, _ = m.action_head.model.blocks[0](x=x.copy())
    assert np.array_equal(out2, np.full((1, 5, 3), 2.0))


def test_get_dit_blocks_missing_raises():
    from layer_skip import get_dit_blocks

    class Empty:
        pass

    try:
        get_dit_blocks(Empty())
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when blocks are missing")


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
