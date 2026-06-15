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

"""Offline tests for capture helpers (numpy + matplotlib only; no torch/GPU/sim)."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from capture import attention_grid_figure, mean_pairwise_cossim  # noqa: E402


def test_cossim_identical_rows_is_one():
    h = np.ones((5, 8), dtype=np.float32) * 3.0
    assert abs(mean_pairwise_cossim(h) - 1.0) < 1e-6


def test_cossim_orthogonal_rows_is_zero():
    h = np.eye(4, dtype=np.float32)  # rows are orthonormal -> pairwise cossim 0
    assert abs(mean_pairwise_cossim(h)) < 1e-6


def test_cossim_opposite_rows_is_negative_one():
    h = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    assert abs(mean_pairwise_cossim(h) - (-1.0)) < 1e-6


def test_cossim_matches_manual_value():
    h = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)  # cossim = 1/sqrt(2)
    assert abs(mean_pairwise_cossim(h) - (1.0 / np.sqrt(2))) < 1e-6


def test_cossim_single_row_is_nan():
    import math

    assert math.isnan(mean_pairwise_cossim(np.zeros((1, 8))))


def test_attention_grid_figure_shape_and_save():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_heads, lq, lk = 24, 12, 20
    attn = np.random.rand(n_heads, lq, lk).astype(np.float32)
    attn /= attn.sum(-1, keepdims=True)  # rows sum to 1 like a real attention map

    fig = attention_grid_figure(attn, "test layer / timestep 0")
    # one Axes per head + padding cells; at least n_heads axes have an image
    n_imgs = sum(len(ax.images) > 0 for ax in fig.axes)
    assert n_imgs == n_heads, n_imgs

    # PdfPages with multiple "timestep" pages works
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "chunk000_layer29.pdf")
        with PdfPages(path) as pdf:
            for _t in range(4):
                f = attention_grid_figure(attn, "page")
                pdf.savefig(f)
                plt.close(f)
        assert os.path.getsize(path) > 0
    plt.close(fig)


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
