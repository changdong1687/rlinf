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
def layer_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Layer-to-layer cosine similarity ``[N, N]`` from per-layer flattened vectors.

    ``vectors[i]`` is layer i's token group flattened to 1-D (e.g. ``vid_i.flatten()``).
    ``M[i, j] = cos(vectors[i], vectors[j])`` — exactly ``F.cosine_similarity`` on the two
    flattened vectors. All vectors must share the same length.
    """
    V = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors])  # [N, D]
    norm = np.linalg.norm(V, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    Vn = V / norm
    return Vn @ Vn.T  # [N, N]


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


def layer_sim_page_figure(mats: dict, suptitle: str):
    """One page: side-by-side N×N layer-similarity heatmaps for vid / act / hidden.

    ``mats``: ordered dict ``{label: [N, N] array}``. Returns the Figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(mats.items())
    fig, axes = plt.subplots(1, len(items), figsize=(5.0 * len(items), 4.6), squeeze=False)
    for ax, (label, m) in zip(axes[0], items):
        m = np.asarray(m, dtype=np.float32)
        im = ax.imshow(m, cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{label}  (N={m.shape[0]})", fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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

        # layer-similarity: per-CURRENT-chunk flattened token vectors per (timestep, layer).
        # {(timestep, layer): {"vid": 1d, "act": 1d, "hid": 1d}}
        self.layer_vecs: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._last_split = None  # (Lq, n_img, n_act) for logging
        # attention for the CURRENT chunk only: {(timestep, layer): np.ndarray[H,Lq,Lk]}
        self.attn_buffer: dict[tuple[int, int], np.ndarray] = {}

        self._handles = []

    # -- lifecycle -------------------------------------------------------------------
    def start_chunk(self):
        self.chunk_idx += 1
        self.timestep_idx = -1
        self.attn_buffer = {}
        self.layer_vecs = {}

    def _new_timestep(self):
        self.timestep_idx += 1

    # -- recording -------------------------------------------------------------------
    def _record_layer_vecs(self, layer: int, x):
        """Stash this layer's flattened token vectors (vid / act / full hidden) for the
        cross-layer similarity matrices built at chunk flush time."""
        h = x[0].detach().float()  # [Lq, d]
        lq = h.shape[0]
        n_act = self.n_act if 0 < self.n_act < lq else max(1, lq // 2)
        n_img = lq - n_act
        self._last_split = (lq, n_img, n_act)
        self.layer_vecs[(self.timestep_idx, layer)] = {
            "vid": h[:n_img].reshape(-1).to(torch.float16).cpu().numpy(),
            "act": h[n_img : n_img + n_act].reshape(-1).to(torch.float16).cpu().numpy(),
            "hid": h.reshape(-1).to(torch.float16).cpu().numpy(),  # full layer output
        }

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
                        self._record_layer_vecs(layer_idx, x)

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

    def flush_chunk_layersim(self, outdir: Path):
        """Build cross-layer similarity matrices for the current chunk and write them.

        One PDF per chunk (``layer_similarity/chunk{c}.pdf``); one page per diffusion
        timestep; each page has 3 N×N heatmaps (video / action / hidden). Raw matrices are
        also saved to a per-chunk .npz.
        """
        if not self.layer_vecs:
            return 0
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        ls_dir = Path(outdir) / "layer_similarity"
        ls_dir.mkdir(parents=True, exist_ok=True)

        timesteps = sorted({t for (t, _l) in self.layer_vecs})
        groups = [("vid", "video tokens"), ("act", "action tokens"), ("hid", "hidden states")]

        npz_store: dict[str, np.ndarray] = {}
        pdf_path = ls_dir / f"chunk{self.chunk_idx:03d}.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            for t in timesteps:
                layers = sorted(layer for (tt, layer) in self.layer_vecs if tt == t)
                mats = {}
                for key, label in groups:
                    vecs = [self.layer_vecs[(t, layer)][key] for layer in layers]
                    mats[label] = layer_similarity_matrix(vecs)  # [N, N]
                    npz_store[f"t{t}_{key}"] = mats[label]
                lq, n_img, n_act = self._last_split or (0, 0, 0)
                suptitle = (
                    f"chunk {self.chunk_idx} | timestep {t} | N={len(layers)} layers | "
                    f"Lq={lq} n_img={n_img} n_act={n_act}"
                )
                fig = layer_sim_page_figure(mats, suptitle)
                pdf.savefig(fig)
                plt.close(fig)
        npz_store["layers"] = np.array(sorted({layer for (_t, layer) in self.layer_vecs}))
        npz_store["timesteps"] = np.array(timesteps)
        np.savez(ls_dir / f"chunk{self.chunk_idx:03d}.npz", **npz_store)

        self.layer_vecs = {}
        return 1
