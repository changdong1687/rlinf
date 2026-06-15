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

from capture import attention_grid_figure, layer_sim_page_figure, layer_similarity_matrix  # noqa: E402


def test_layer_sim_diagonal_is_one():
    # Each layer's flattened vector vs itself -> 1 on the diagonal.
    vecs = [np.random.rand(12 * 8).astype(np.float32) for _ in range(5)]
    m = layer_similarity_matrix(vecs)
    assert m.shape == (5, 5)
    assert np.allclose(np.diag(m), 1.0, atol=1e-6)


def test_layer_sim_symmetric():
    vecs = [np.random.rand(20).astype(np.float32) for _ in range(4)]
    m = layer_similarity_matrix(vecs)
    assert np.allclose(m, m.T, atol=1e-6)


def test_layer_sim_identical_layers_all_ones():
    v = np.random.rand(30).astype(np.float32)
    m = layer_similarity_matrix([v, v.copy(), v.copy()])
    assert np.allclose(m, 1.0, atol=1e-6)


def test_layer_sim_opposite_is_negative_one():
    v = np.random.rand(16).astype(np.float32)
    m = layer_similarity_matrix([v, -v])
    assert abs(m[0, 1] - (-1.0)) < 1e-6


def test_layer_sim_matches_flatten_cosine():
    # Matches F.cosine_similarity on the flattened [n_tok, d] tensors.
    a = np.random.rand(4, 8).astype(np.float64)
    b = np.random.rand(4, 8).astype(np.float64)
    m = layer_similarity_matrix([a.reshape(-1), b.reshape(-1)])
    af, bf = a.reshape(-1), b.reshape(-1)
    expected = float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf)))
    assert abs(m[0, 1] - expected) < 1e-9


def test_layer_sim_page_figure_three_panels():
    import matplotlib.pyplot as plt

    n = 6
    mats = {
        "video tokens": np.random.rand(n, n).astype(np.float32),
        "action tokens": np.random.rand(n, n).astype(np.float32),
        "hidden states": np.random.rand(n, n).astype(np.float32),
    }
    fig = layer_sim_page_figure(mats, "chunk 0 | timestep 0")
    n_imgs = sum(len(ax.images) > 0 for ax in fig.axes)
    assert n_imgs == 3, n_imgs
    plt.close(fig)


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
