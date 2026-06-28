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

"""Shared model-loading + websocket-serving for the cache-method baselines.

This is a trimmed adaptation of
``examples/embodiment/dreamzero_libero_eval/policy_server.py`` (same RLinf model
construction, same LIBERO observation preprocessing, same pickle-over-websocket
protocol, so the *existing* ``libero_eval_client.py`` works unchanged against any
of these servers). The only differences:

* no ``--layer-skip`` flag (this folder is about temporal caches, not layer skip);
* :func:`add_common_model_args` / :func:`build_model` are reusable so each method
  server (teacache/bac/bwcache/baseline) is a thin entry script that just adds its
  own cache flags, calls :func:`cache_common.force_full_compute_env`, builds the
  model, installs its cache, and serves.

Keeping model loading in one place (rather than copy-pasted four times) avoids the
frozen-component / precision / checkpoint-overlay logic drifting between servers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import os
import pickle
import sys
import traceback

import numpy as np
import torch
import websockets
import websockets.asyncio.server
import websockets.frames
from omegaconf import OmegaConf, open_dict

# Make this folder importable (cache_common, *_dreamzero) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlinf.envs.libero.utils import (  # noqa: E402
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.models.embodiment.dreamzero import get_model  # noqa: E402

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("dreamzero_cache_eval.server")

_PRECISION_TO_DTYPE = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}

_FROZEN_COMPONENT_MARKERS = ("umt5", "open-clip", "vae")


@dataclasses.dataclass
class ServerMetadata:
    embodiment: str = "libero_sim"
    action_space: str = "osc_pose"
    action_dim: int = 7
    num_action_chunks: int = 16
    num_layers: int = 0
    # Cache experiment bookkeeping (recorded in the client's results.json):
    cache_method: str = "none"
    cache_config: str = ""  # human-readable params, e.g. "thresh=0.15"


class LiberoRLinfPolicy:
    """Raw LIBERO obs -> RLinf ``env_obs`` -> ``predict_action_batch`` chunk.

    Identical preprocessing to the dreamzero_libero_eval server (180-degree image
    rotation via ``get_libero_image`` and the eef_pos+axis_angle+gripper state
    layout), so inference matches RLinf training / the built-in eval exactly.
    """

    def __init__(self, model, device: torch.device) -> None:
        self._model = model
        self._device = device

    def reset(self, payload: dict) -> None:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_env_obs(self, obs: dict) -> dict:
        main_image = get_libero_image(obs)
        wrist_image = get_libero_wrist_image(obs)
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            ]
        ).astype(np.float32)
        return {
            "main_images": np.ascontiguousarray(main_image)[None, ...],
            "wrist_images": np.ascontiguousarray(wrist_image)[None, ...],
            "states": state[None, ...],
            "task_descriptions": [str(obs.get("prompt", ""))],
        }

    def infer(self, obs: dict) -> dict:
        env_obs = self._build_env_obs(obs)
        with torch.no_grad():
            actions, _ = self._model.predict_action_batch(env_obs, mode="eval")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        return {"actions": actions[0]}


class PicklePolicyServer:
    def __init__(self, policy, metadata, host, port) -> None:
        self._policy = policy
        self._metadata = metadata
        self._host = host
        self._port = port

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        logger.info("Listening on ws://%s:%d", self._host, self._port)
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(pickle.dumps(dataclasses.asdict(self._metadata)))
        while True:
            try:
                payload = pickle.loads(await websocket.recv())
                endpoint = payload.pop("endpoint")
                if endpoint == "reset":
                    await asyncio.to_thread(self._policy.reset, payload)
                    await websocket.send(pickle.dumps({"status": "reset successful"}))
                elif endpoint == "infer":
                    action = await asyncio.to_thread(self._policy.infer, payload)
                    await websocket.send(pickle.dumps(action))
                else:
                    raise ValueError(f"Unsupported endpoint: {endpoint}")
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


@contextlib.contextmanager
def _skip_frozen_component_loads():
    """Skip the ~10GB text/CLIP/VAE .pth reads when a full --model-path is used."""
    orig_torch_load = torch.load
    orig_load_sd = torch.nn.Module.load_state_dict

    def _patched_load(f, *a, **k):
        name = str(f).lower()
        if name.endswith(".pth") and any(m in name for m in _FROZEN_COMPONENT_MARKERS):
            logger.info("skip-frozen-component-load: skipping %s", f)
            return {}
        return orig_torch_load(f, *a, **k)

    def _patched_load_sd(self, state_dict, strict=True, *a, **k):
        if isinstance(state_dict, dict) and len(state_dict) == 0:
            return torch.nn.modules.module._IncompatibleKeys([], [])
        return orig_load_sd(self, state_dict, strict=strict, *a, **k)

    torch.load = _patched_load
    torch.nn.Module.load_state_dict = _patched_load_sd
    try:
        yield
    finally:
        torch.load = orig_torch_load
        torch.nn.Module.load_state_dict = orig_load_sd


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """Register the model-loading / server flags shared by all cache servers."""
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config", type=str, default=os.path.join(here, "config", "dreamzero_5b_libero.yaml"))
    parser.add_argument("--model-path", type=str, default=None, help="Full DreamZero checkpoint dir (safetensors).")
    parser.add_argument("--ckpt-path", type=str, default=None, help="RLinf full_weights.pt state dict to overlay.")
    parser.add_argument("--metadata-json-path", type=str, default=None, help="q99 normalization stats (MUST match SFT).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help=(
            "Model dtype: 'bf16' (default; faster/less memory), 'fp32', 'fp16', or "
            "'native' (no cast -> DiT stays float32 = RLinf rollout = best accuracy). "
            "NOTE: bf16 trades accuracy for speed (upstream LIBERO ~96.7 -> ~79)."
        ),
    )
    parser.add_argument("--skip-frozen-component-load", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stats-out", type=str, default=None, help="Write per-method DiT compute/timing stats JSON here.")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--diffusion-model-pretrained-path", type=str, default=None)
    parser.add_argument("--image-encoder-pretrained-path", type=str, default=None)
    parser.add_argument("--text-encoder-pretrained-path", type=str, default=None)
    parser.add_argument("--vae-pretrained-path", type=str, default=None)


def build_model(args: argparse.Namespace, device: torch.device):
    """Load actor.model config, build the DreamZero model, overlay the checkpoint.

    Returns ``(model, num_action_chunks)``. Precision: ``--precision`` defaults to
    ``bf16`` (cast the whole model); pass ``--precision native`` to keep the
    constructed dtype (DiT float32), which matches RLinf's rollout and gives the
    best LIBERO accuracy (bf16 measurably degrades it; see the --precision help).
    """
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.actor.model
    prec = (args.precision or "native").lower()
    with open_dict(model_cfg):
        if prec not in ("native", "none"):
            model_cfg.precision = prec
        if args.metadata_json_path is not None:
            model_cfg.metadata_json_path = args.metadata_json_path
        if args.tokenizer_path is not None:
            model_cfg.tokenizer_path = args.tokenizer_path
        if args.diffusion_model_pretrained_path is not None:
            model_cfg.diffusion_model_pretrained_path = args.diffusion_model_pretrained_path
        if args.image_encoder_pretrained_path is not None:
            model_cfg.image_encoder_pretrained_path = args.image_encoder_pretrained_path
        if args.text_encoder_pretrained_path is not None:
            model_cfg.text_encoder_pretrained_path = args.text_encoder_pretrained_path
        if args.vae_pretrained_path is not None:
            model_cfg.vae_pretrained_path = args.vae_pretrained_path
        model_cfg.model_path = args.model_path if args.model_path else None

    # Default is bf16 (cast the whole model). Pass --precision native to keep the
    # model's constructed dtype (DiT float32), which matches RLinf's rollout.
    if prec in ("native", "none"):
        torch_dtype = None
        logger.info("precision=native: no cast (DiT stays float32, matches RLinf rollout / best accuracy)")
    elif prec in _PRECISION_TO_DTYPE:
        torch_dtype = _PRECISION_TO_DTYPE[prec]
        if torch_dtype == torch.bfloat16:
            logger.warning(
                "precision=bf16: faster / less memory, but the upstream DreamZero eval "
                "reports LIBERO-Spatial success drops from ~96.7%% to ~79%% vs native. "
                "Use --precision native for the accurate reference."
            )
    else:
        raise ValueError(f"Unknown --precision {args.precision!r}; choose bf16 / fp32 / fp16 / native.")

    logger.info("Building DreamZero model (dtype=%s, model_path=%s)", torch_dtype, model_cfg.model_path)
    skip_components = bool(args.skip_frozen_component_load) and bool(args.model_path)
    cm = _skip_frozen_component_loads() if skip_components else contextlib.nullcontext()
    with cm:
        model = get_model(model_cfg, torch_dtype=torch_dtype)

    if args.ckpt_path:
        logger.info("Overlaying trained checkpoint: %s", args.ckpt_path)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("load_state_dict missing keys: %d (e.g. %s)", len(missing), missing[:5])
        if unexpected:
            logger.warning("load_state_dict unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:5])
    elif not args.model_path:
        logger.warning("Neither --model-path nor --ckpt-path given: results will be meaningless.")

    model.eval()
    model.to(device)
    num_action_chunks = int(model_cfg.get("num_action_chunks", 16))
    return model, num_action_chunks


def resolve_device(args: argparse.Namespace) -> torch.device:
    return torch.device(args.device if torch.cuda.is_available() else "cpu")


def serve(model, num_action_chunks, device, args, cache_method: str, cache_config: str, num_layers: int = 0) -> None:
    """Wrap the model in the policy + websocket server and run forever."""
    metadata = ServerMetadata(
        num_action_chunks=num_action_chunks,
        num_layers=num_layers,
        cache_method=cache_method,
        cache_config=cache_config,
    )
    policy = LiberoRLinfPolicy(model, device)
    logger.info(
        "Model ready. method=%s config=[%s] num_action_chunks=%d num_layers=%d",
        cache_method,
        cache_config,
        num_action_chunks,
        num_layers,
    )
    PicklePolicyServer(policy, metadata, host=args.host, port=args.port).serve_forever()
