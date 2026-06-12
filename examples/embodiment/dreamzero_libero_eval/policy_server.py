#!/usr/bin/env python3
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

"""Serve an RLinf-trained DreamZero policy over websockets for LIBERO evaluation.

Unlike ``../DreamZero-Libero/eval_utils/run_libero_server.py`` (which loads the model
via groot's ``GrootSimPolicy``), this server reuses RLinf's own model construction and
inference path, so preprocessing / normalization / action conventions match RLinf
training and the official ``libero_spatial_eval_dreamzero`` eval exactly.

Model loading mirrors ``rlinf/workers/rollout/hf/huggingface_worker.py::init_worker``:

    model = get_model(actor_model_cfg, torch_dtype=bf16)   # build from Wan components
    model.load_state_dict(torch.load(ckpt_path))           # overlay trained weights
    model.eval().to(device)

Inference is stateless per call: ``DreamZeroPolicy.predict_action_batch`` returns a full
``[B, num_action_chunks, 7]`` action chunk from the current observation (no autoregressive
cache across calls), so ``reset`` is a no-op for the model.

Protocol: pickle-over-websocket (matches the reference run_libero_server / run_libero_eval).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import pickle
import traceback

import numpy as np
import torch
import websockets
import websockets.asyncio.server
import websockets.frames
from omegaconf import OmegaConf, open_dict

# RLinf model construction + inference.
from rlinf.models.embodiment.dreamzero import get_model

# Reuse RLinf's exact LIBERO observation preprocessing so server-side conversion is
# pixel-identical to training (rlinf/envs/libero/libero_env.py::_extract_image_and_state).
from rlinf.envs.libero.utils import (
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("dreamzero_libero_server")

_PRECISION_TO_DTYPE = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


@dataclasses.dataclass
class ServerMetadata:
    embodiment: str = "libero_sim"
    action_space: str = "osc_pose"
    action_dim: int = 7
    num_action_chunks: int = 16


class LiberoRLinfPolicy:
    """Thin wrapper that turns raw LIBERO observations into RLinf ``env_obs`` and runs inference."""

    def __init__(self, model, device: torch.device) -> None:
        self._model = model
        self._device = device

    def reset(self, payload: dict) -> None:
        # RLinf DreamZero inference is stateless per chunk: nothing to reset on the model.
        # Free any fragmented cache between episodes to keep memory stable.
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_env_obs(self, obs: dict) -> dict:
        """Map robosuite-style LIBERO obs to RLinf rollout ``env_obs`` (batch size 1).

        Mirrors ``LiberoEnv._extract_image_and_state`` / ``_wrap_obs``:
          - images go through the 180-degree rotation in ``get_libero_image`` /
            ``get_libero_wrist_image`` (matches train preprocessing).
          - state = eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2).
        """
        main_image = get_libero_image(obs)  # [H, W, 3] uint8, rotated 180 deg
        wrist_image = get_libero_wrist_image(obs)  # [H, W, 3] uint8, rotated 180 deg

        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            ]
        ).astype(np.float32)

        return {
            "main_images": np.ascontiguousarray(main_image)[None, ...],  # [1, H, W, 3]
            "wrist_images": np.ascontiguousarray(wrist_image)[None, ...],  # [1, H, W, 3]
            "states": state[None, ...],  # [1, 8]
            "task_descriptions": [str(obs.get("prompt", ""))],
        }

    def infer(self, obs: dict) -> dict:
        env_obs = self._build_env_obs(obs)
        with torch.no_grad():
            actions, _ = self._model.predict_action_batch(env_obs, mode="eval")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        # actions: [B=1, num_action_chunks, action_dim]; return the single-env chunk.
        return {"actions": actions[0]}


class PicklePolicyServer:
    """Single-GPU pickle-over-websocket policy server."""

    def __init__(
        self,
        policy: LiberoRLinfPolicy,
        metadata: ServerMetadata,
        host: str,
        port: int,
    ) -> None:
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
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(pickle.dumps(dataclasses.asdict(self._metadata)))
        while True:
            try:
                payload = pickle.loads(await websocket.recv())
                endpoint = payload.pop("endpoint")
                if endpoint == "reset":
                    self._policy.reset(payload)
                    await websocket.send(pickle.dumps({"status": "reset successful"}))
                elif endpoint == "infer":
                    action = self._policy.infer(payload)
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


def _build_model(args: argparse.Namespace, device: torch.device):
    """Load actor.model config, build the DreamZero model, and overlay the checkpoint."""
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.actor.model

    with open_dict(model_cfg):
        if args.precision is not None:
            model_cfg.precision = args.precision
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
        # Two ways to provide trained weights:
        #   --model-path DIR : a full DreamZero checkpoint dir (model.safetensors[.index.json]
        #                      + config.json + experiment_cfg/). get_model loads it natively
        #                      and skips Wan component loading (full weights already include them).
        #   --ckpt-path .pt  : an RLinf full_weights.pt state dict, overlaid after get_model.
        model_cfg.model_path = args.model_path if args.model_path else None

    if args.model_path and args.ckpt_path:
        logger.warning(
            "Both --model-path and --ckpt-path given; --model-path loads the full "
            "safetensors and --ckpt-path will overlay on top of it."
        )

    precision = str(model_cfg.get("precision", "bf16"))
    torch_dtype = _PRECISION_TO_DTYPE.get(precision, torch.bfloat16)

    logger.info(
        "Building DreamZero model (precision=%s, dtype=%s, model_path=%s)",
        precision,
        torch_dtype,
        model_cfg.model_path,
    )
    # When model_path is set, get_model loads the full safetensors shards from that dir.
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
            logger.warning(
                "load_state_dict unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:5]
            )
    elif not args.model_path:
        logger.warning(
            "Neither --model-path nor --ckpt-path given: serving component-initialized "
            "weights only (untrained action head). Results will be meaningless."
        )

    model.eval()
    model.to(device)
    num_action_chunks = int(model_cfg.get("num_action_chunks", 16))
    return model, num_action_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Standalone actor.model YAML. Defaults to config/dreamzero_5b_libero.yaml next to this script.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Full DreamZero checkpoint DIRECTORY (model.safetensors[.index.json] + config.json "
            "+ experiment_cfg/), e.g. the open RLinf-DreamZero-...-SFT-Step18000 download. "
            "Loaded natively by get_model; Wan component paths are not needed in this mode."
        ),
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help=(
            "RLinf full_weights.pt state dict (alternative to --model-path). Overlaid on top "
            "of the built model. If both are omitted, only component weights are used."
        ),
    )
    parser.add_argument(
        "--metadata-json-path",
        type=str,
        default=None,
        help="metadata.json with q99 normalization stats (MUST match SFT). Overrides the YAML value.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--precision", type=str, default=None, help="Override model precision (e.g. bf16, fp32).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    # Optional pretrained component path overrides (else taken from YAML / HF cache).
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--diffusion-model-pretrained-path", type=str, default=None)
    parser.add_argument("--image-encoder-pretrained-path", type=str, default=None)
    parser.add_argument("--text-encoder-pretrained-path", type=str, default=None)
    parser.add_argument("--vae-pretrained-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    import os

    args = parse_args()
    if args.config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(here, "config", "dreamzero_5b_libero.yaml")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, num_action_chunks = _build_model(args, device)

    metadata = ServerMetadata(num_action_chunks=num_action_chunks)
    policy = LiberoRLinfPolicy(model, device)
    logger.info("Model ready. action_dim=%d num_action_chunks=%d", metadata.action_dim, num_action_chunks)

    server = PicklePolicyServer(policy, metadata, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
