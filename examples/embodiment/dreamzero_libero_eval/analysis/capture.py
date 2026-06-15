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

"""DreamZero DiT inference capture: token-token cosine similarity + attention maps.

Runtime instrumentation (no groot edits), in the spirit of layer_skip.py:

- Hidden states: a forward hook on each ``CausalWanAttentionBlock`` captures its output
  ``x`` ``[B, Lq, dim]``. During AR inference the rows are ``[image tokens | action tokens]``
  (no clean/state doubling — that only happens in the teacher-forcing path). We compute
  three mean pairwise cosine similarities per (chunk, diffusion-timestep, layer):
  video-only, action-only, and combined.

- Attention maps: during AR inference each layer calls ``AttentionModule.forward(q, k, v)``
  exactly once (see CausalWanSelfAttention.forward kv-cache branch), with
  ``q=[image_q; action_q]`` and ``k=[cached_k; image_k; action_k]``. We wrap that module to
  recompute ``softmax(q·kᵀ/√d)`` per head and store ``[H, Lq, Lk]`` — the real (possibly
  non-square) AR attention. Output of the original module is returned untouched.

Diffusion timestep tracking: a forward-pre-hook on block 0 bumps the timestep counter
(every full 30-layer pass == one denoising step). ``start_chunk`` resets it per chunk.

The helper math (``mean_pairwise_cossim``) and the PDF page builder are pure functions so
they can be unit-tested offline (numpy/matplotlib, no GPU).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:  # torch is only needed when actually running capture on a model
    import torch
except Exception:  # pragma: no cover - offline tests stub this out
    torch = None


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# --------------------------------------------------------------------------------------
def mean_pairwise_cossim(h: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity among rows of ``h`` ``[N, d]``.

    Returns NaN when there are fewer than 2 rows.
    """
    h = np.asarray(h, dtype=np.float64)
    n = h.shape[0]
    if n < 2:
        return float("nan")
    norm = np.linalg.norm(h, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    hn = h / norm
    sim = hn @ hn.T  # [N, N]
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def attention_grid_figure(attn_heads: np.ndarray, title: str):
    """Build a matplotlib Figure: one subplot per head, imshow of [Lq, Lk].

    ``attn_heads``: ``[H, Lq, Lk]``. Returns the Figure (caller saves/closes it).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attn_heads = np.asarray(attn_heads, dtype=np.float32)
    h = attn_heads.shape[0]
    ncols = int(math.ceil(math.sqrt(h)))
    nrows = int(math.ceil(h / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.4), squeeze=False)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx < h:
            ax.imshow(attn_heads[idx], aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_title(f"head {idx}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------------------
# Capture context + installation
# --------------------------------------------------------------------------------------
class CaptureContext:
    def __init__(self, want_cossim=True, want_attn=True, layers=None, n_act=16):
        self.want_cossim = want_cossim
        self.want_attn = want_attn
        self.layers = set(layers) if layers is not None else None  # None = all layers
        self.n_act = int(n_act)
        self.enabled = False

        self.num_layers = 0
        self.chunk_idx = -1
        self.timestep_idx = -1

        # cossim: flat list of records (small, kept for the whole run)
        self.cossim_records: list[dict] = []
        # attention for the CURRENT chunk only: {(timestep, layer): np.ndarray[H,Lq,Lk]}
        self.attn_buffer: dict[tuple[int, int], np.ndarray] = {}

        self._handles = []

    # -- lifecycle -------------------------------------------------------------------
    def start_chunk(self):
        self.chunk_idx += 1
        self.timestep_idx = -1
        self.attn_buffer = {}

    def _new_timestep(self):
        self.timestep_idx += 1

    # -- recording -------------------------------------------------------------------
    def _record_cossim(self, layer: int, x):
        h = x[0].detach().float().cpu().numpy()  # [Lq, d]
        lq = h.shape[0]
        n_act = self.n_act if 0 < self.n_act < lq else max(1, lq // 2)
        n_img = lq - n_act
        self.cossim_records.append(
            {
                "chunk": self.chunk_idx,
                "timestep": self.timestep_idx,
                "layer": layer,
                "Lq": lq,
                "n_img": n_img,
                "n_act": n_act,
                "vid": mean_pairwise_cossim(h[:n_img]),
                "act": mean_pairwise_cossim(h[n_img : n_img + n_act]),
                "comb": mean_pairwise_cossim(h[: n_img + n_act]),
            }
        )

    def _record_attention(self, layer: int, causal: bool, q, k):
        qf = q[0].detach().float()  # [Lq, H, d]
        kf = k[0].detach().float()  # [Lk, H, d]
        d = qf.shape[-1]
        scale = 1.0 / math.sqrt(d)
        scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale  # [H, Lq, Lk]
        if causal:
            lq, lk = scores.shape[1], scores.shape[2]
            mask = torch.ones(lq, lk, dtype=torch.bool, device=scores.device).tril(lk - lq)
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1).to(torch.float16).cpu().numpy()
        key = (self.timestep_idx, layer)
        prev = self.attn_buffer.get(key)
        # Inference does one attn call per layer; if several, keep the widest-query one.
        if prev is None or attn.shape[1] >= prev.shape[1]:
            self.attn_buffer[key] = attn

    # -- install / remove ------------------------------------------------------------
    def install(self, model):
        dit = getattr(getattr(model, "action_head", None), "model", None)
        if dit is None or not hasattr(dit, "blocks"):
            raise RuntimeError("Could not locate model.action_head.model.blocks")
        blocks = dit.blocks
        self.num_layers = len(blocks)
        try:
            self.n_act = int(model.action_head.action_horizon)
        except Exception:
            pass

        def pre_hook(_mod, _args):
            if self.enabled:
                self._new_timestep()

        self._handles.append(blocks[0].register_forward_pre_hook(pre_hook))

        for layer, blk in enumerate(blocks):

            def make_fwd_hook(layer_idx):
                def hook(_mod, _inp, output):
                    if self.enabled and self.want_cossim:
                        x = output[0] if isinstance(output, (tuple, list)) else output
                        self._record_cossim(layer_idx, x)

                return hook

            self._handles.append(blk.register_forward_hook(make_fwd_hook(layer)))

            sa = getattr(blk, "self_attn", None)
            if sa is None:
                continue
            for attr, causal in (("attn", False), ("causal_attn", True)):
                am = getattr(sa, attr, None)
                if am is None:
                    continue
                self._wrap_attention_module(am, layer, causal)

        return self

    def _wrap_attention_module(self, am, layer: int, causal: bool):
        if getattr(am, "_capture_wrapped", False):
            am._capture_layer = layer  # shared instance? keep latest
            return
        orig_forward = am.forward
        am._capture_layer = layer
        am._capture_causal = causal
        am._capture_wrapped = True

        def wrapper(q, k, v, *args, **kwargs):
            out = orig_forward(q, k, v, *args, **kwargs)
            if self.enabled and self.want_attn:
                lyr = getattr(am, "_capture_layer", layer)
                if self.layers is None or lyr in self.layers:
                    try:
                        self._record_attention(lyr, getattr(am, "_capture_causal", causal), q, k)
                    except Exception:
                        pass
            return out

        am.forward = wrapper

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    # -- rendering -------------------------------------------------------------------
    def flush_chunk_attention(self, outdir: Path):
        """Write the current chunk's attention to PDFs (one per (chunk, layer)) and clear."""
        if not self.attn_buffer:
            return 0
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        attn_dir = Path(outdir) / "attention"
        attn_dir.mkdir(parents=True, exist_ok=True)

        layers = sorted({layer for (_t, layer) in self.attn_buffer})
        n_written = 0
        for layer in layers:
            timesteps = sorted(t for (t, lyr) in self.attn_buffer if lyr == layer)
            pdf_path = attn_dir / f"chunk{self.chunk_idx:03d}_layer{layer:02d}.pdf"
            with PdfPages(str(pdf_path)) as pdf:
                for t in timesteps:
                    attn = self.attn_buffer[(t, layer)]  # [H, Lq, Lk]
                    title = f"chunk {self.chunk_idx} | layer {layer} | timestep {t} | {attn.shape[0]} heads | Lq={attn.shape[1]} Lk={attn.shape[2]}"
                    fig = attention_grid_figure(attn, title)
                    pdf.savefig(fig)
                    plt.close(fig)
            n_written += 1
        self.attn_buffer = {}
        return n_written

    def render_cossim(self, outdir: Path):
        """Save cossim records (npz + csv) and per-group layer curves (one line per timestep)."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if not self.cossim_records:
            return

        import csv

        keys = ["chunk", "timestep", "layer", "Lq", "n_img", "n_act", "vid", "act", "comb"]
        with open(outdir / "cossim.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.cossim_records)
        np.savez(
            outdir / "cossim.npz",
            **{k: np.array([r[k] for r in self.cossim_records]) for k in keys},
        )

        self._plot_cossim_curves(outdir)

    def _plot_cossim_curves(self, outdir: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recs = self.cossim_records
        layers = sorted({r["layer"] for r in recs})
        timesteps = sorted({r["timestep"] for r in recs})
        for group, label in (("vid", "video tokens"), ("act", "action tokens"), ("comb", "video+action")):
            fig, ax = plt.subplots(figsize=(8, 5))
            for t in timesteps:
                ys = []
                for layer in layers:
                    vals = [r[group] for r in recs if r["layer"] == layer and r["timestep"] == t]
                    vals = [v for v in vals if not math.isnan(v)]
                    ys.append(np.mean(vals) if vals else math.nan)
                ax.plot(layers, ys, marker="o", markersize=3, label=f"timestep {t}")
            ax.set_xlabel("layer")
            ax.set_ylabel("mean pairwise cosine similarity")
            ax.set_title(f"Token-token cossim across layers — {label}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"cossim_{group}.png", dpi=150)
            plt.close(fig)
