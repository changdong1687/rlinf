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

"""Run official LIBERO benchmark rollouts against a remote RLinf DreamZero policy server.

This client owns the LIBERO simulator (``OffScreenRenderEnv``) and talks to
``policy_server.py`` over a pickle websocket. It sends raw robosuite-style observations;
the server does all RLinf-specific preprocessing and returns a ``[num_action_chunks, 7]``
action chunk that is executed directly on the environment.

The gripper dimension is already binarized to +-1 by RLinf's ``predict_action_batch``
(``binarize_gripper=True``) and ``LiberoEnv.step`` passes actions through untouched, so we
do NOT apply any extra gripper transform here (unlike the groot reference client).

Default ``--open-loop-horizon`` is the full chunk length (execute all predicted actions,
then re-query), matching RLinf's own eval loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np

# NOTE: torch / imageio / tqdm / websockets are imported lazily inside the functions that
# use them. This keeps ``DreamZeroLiberoClient`` (the chunk-execution logic) importable with
# only numpy, so it can be unit-tested without the simulator / GPU runtime installed.

# Default to a LIBERO checkout sitting next to the RLinf repo (parents[3] is the repo root).
# run_client.sh always passes --libero-root explicitly, so this is only a fallback.
DEFAULT_LIBERO_ROOT = Path(__file__).resolve().parents[3].parent / "LIBERO"


class PickleWebsocketClient:
    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        import websockets.sync.client

        self._uri = f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=None,
        )
        self._metadata = pickle.loads(self._ws.recv())

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        obs["endpoint"] = "infer"
        self._ws.send(pickle.dumps(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return pickle.loads(response)

    def reset(self, reset_info: dict | None = None) -> None:
        payload = {} if reset_info is None else dict(reset_info)
        payload["endpoint"] = "reset"
        self._ws.send(pickle.dumps(payload))
        self._ws.recv()


class DreamZeroLiberoClient:
    """Closed-loop driver: caches a predicted chunk and re-queries every ``open_loop_horizon`` steps."""

    def __init__(
        self,
        host: str,
        port: int,
        open_loop_horizon: int | None = None,
        transport=None,
    ) -> None:
        # ``transport`` is any object exposing ``infer(dict)->dict`` / ``reset(dict)`` /
        # ``metadata`` (default: a real pickle websocket client). Injectable for testing.
        self.client = transport if transport is not None else PickleWebsocketClient(host=host, port=port)
        # None -> execute the full returned chunk before re-querying (RLinf eval behavior).
        self.open_loop_horizon = open_loop_horizon
        self.session_id = str(uuid.uuid4())
        self.pred_action_chunk: np.ndarray | None = None
        self.actions_from_chunk_completed = 0

    def reset(self) -> None:
        self.client.reset({"session_id": self.session_id})
        self.session_id = str(uuid.uuid4())
        self.pred_action_chunk = None
        self.actions_from_chunk_completed = 0

    def _horizon(self, chunk_len: int) -> int:
        if self.open_loop_horizon is None or self.open_loop_horizon <= 0:
            return chunk_len
        return min(self.open_loop_horizon, chunk_len)

    def infer(self, obs: dict, instruction: str) -> np.ndarray:
        needs_new_chunk = (
            self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= self._horizon(len(self.pred_action_chunk))
        )
        if needs_new_chunk:
            request = {
                "agentview_image": np.asarray(obs["agentview_image"], dtype=np.uint8),
                "robot0_eye_in_hand_image": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
                "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
                "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }
            result = self.client.infer(request)
            actions = result["actions"] if isinstance(result, dict) else result
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"Expected action chunk of shape (N, 7), got {actions.shape}")
            self.pred_action_chunk = actions
            self.actions_from_chunk_completed = 0

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1
        return action


def ensure_libero_imports(libero_root: Path) -> None:
    libero_root = libero_root.resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def load_init_states(init_states_path: Path):
    """Load LIBERO init states compatibly across PyTorch versions."""
    import torch

    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        return torch.load(init_states_path)


def make_rollout_frame(obs: dict) -> np.ndarray:
    # LIBERO render frames are vertically inverted; flip only for saved videos.
    agentview = np.flipud(np.asarray(obs["agentview_image"], dtype=np.uint8)).copy()
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)).copy()
    if wrist.shape[0] != agentview.shape[0]:
        row_idx = np.linspace(0, wrist.shape[0] - 1, agentview.shape[0]).astype(np.int64)
        target_width = max(1, int(round(wrist.shape[1] * agentview.shape[0] / wrist.shape[0])))
        col_idx = np.linspace(0, wrist.shape[1] - 1, target_width).astype(np.int64)
        wrist = wrist[row_idx][:, col_idx]
    return np.concatenate([agentview, wrist], axis=1)


def write_rollout_video(frames: list[np.ndarray], output_path: Path, fps: int = 20) -> None:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")


def write_results(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["task_id", "task_name", "language", "success_rate"])
        writer.writeheader()
        for task in results["tasks"]:
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "language": task["language"],
                    "success_rate": task["success_rate"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="localhost", help="DreamZero policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="DreamZero policy server port.")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="Path to the local LIBERO repo.")
    parser.add_argument("--benchmark-name", type=str, default="libero_spatial", help="LIBERO benchmark name.")
    parser.add_argument("--task-order-index", type=int, default=0, help="Task order index for 10-task suites.")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None, help="Optional task ids to evaluate.")
    parser.add_argument("--n-eval", type=int, default=20, help="Episodes per task.")
    parser.add_argument("--max-steps", type=int, default=480, help="Maximum rollout length (matches RLinf libero eval).")
    parser.add_argument(
        "--num-settle-steps",
        type=int,
        default=15,
        help="Zero-action (gripper open) steps after set_init_state, before the policy "
        "starts. MUST match RLinf LiberoEnv.reset (15). Lower values start the policy on "
        "an unsettled, out-of-distribution scene and reduce success.",
    )
    parser.add_argument("--camera-height", type=int, default=256, help="Render camera height (RLinf trains at 256).")
    parser.add_argument("--camera-width", type=int, default=256, help="Render camera width (RLinf trains at 256).")
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=-1,
        help="Actions to execute per server call. <=0 means execute the full chunk (RLinf eval behavior).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/libero_eval"), help="Directory for JSON/CSV results.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional checkpoint path recorded in results.json.")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos for the first few episodes per task.")
    parser.add_argument("--video-episodes-per-task", type=int, default=1, help="Episodes per task to save when --save-video.")
    parser.add_argument("--seed", type=int, default=0, help="Numpy seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    import torch
    from tqdm.auto import tqdm

    args = parse_args()
    np.random.seed(args.seed)
    ensure_libero_imports(args.libero_root)

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    open_loop_horizon = None if args.open_loop_horizon <= 0 else args.open_loop_horizon

    benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
    task_ids = args.task_ids if args.task_ids is not None else list(range(benchmark.n_tasks))
    client = DreamZeroLiberoClient(args.host, args.port, open_loop_horizon=open_loop_horizon)

    tqdm.write(
        f"[eval][start] benchmark={args.benchmark_name} task_ids={task_ids} n_eval={args.n_eval} "
        f"max_steps={args.max_steps} open_loop_horizon={open_loop_horizon} "
        f"server_metadata={client.client.metadata}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark_name": args.benchmark_name,
        "task_order_index": args.task_order_index,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "open_loop_horizon": open_loop_horizon,
        "num_settle_steps": args.num_settle_steps,
        "checkpoint_path": str(args.checkpoint_path.resolve()) if args.checkpoint_path is not None else None,
        "server_metadata": client.client.metadata,
        "tasks": [],
        "mean_success_rate": 0.0,
    }

    episode_idx = 0

    task_progress = tqdm(task_ids, desc="Tasks", unit="task")
    for task_id in task_progress:
        task = benchmark.get_task(task_id)
        task_progress.set_postfix(task_id=task_id, task_name=task.name[:32], refresh=False)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=args.camera_height,
            camera_widths=args.camera_width,
        )
        init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = load_init_states(init_states_path)

        successes = 0
        task_result = {
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "success_rate": 0.0,
            "episodes": [],
        }
        results["tasks"].append(task_result)

        episode_progress = tqdm(range(args.n_eval), desc=f"Task {task_id} Episodes", unit="ep", leave=False)
        for episode_idx in episode_progress:
            client.reset()
            env.reset()
            init_state = init_states[episode_idx % len(init_states)]
            if torch.is_tensor(init_state):
                init_state = init_state.cpu().numpy()
            obs = env.set_init_state(init_state)

            # Let the scene settle BEFORE the policy starts. This must match RLinf's
            # LiberoEnv.reset (rlinf/envs/libero/libero_env.py): it steps the env
            # `num_settle_steps` times with a zero action whose gripper dim is -1 (open).
            # Mismatching this (e.g. 5 steps / gripper 0) starts the policy on an
            # unsettled, out-of-distribution observation and lowers success.
            settle_action = np.zeros(7, dtype=np.float32)
            settle_action[-1] = -1.0  # gripper open, matches reset_gripper_open=True
            for _ in range(args.num_settle_steps):
                obs, _, _, _ = env.step(settle_action)

            success = False
            steps = 0
            save_video = args.save_video and episode_idx < args.video_episodes_per_task
            video_frames = [make_rollout_frame(obs)] if save_video else []

            rollout_progress = tqdm(total=args.max_steps, desc=f"Task {task_id} Ep {episode_idx}", unit="step", leave=False)
            while steps < args.max_steps:
                action = client.infer(obs, task.language)
                obs, _, _, _ = env.step(action)
                steps += 1
                rollout_progress.update(1)
                if save_video:
                    video_frames.append(make_rollout_frame(obs))
                if env.check_success():
                    success = True
                    break
            rollout_progress.set_postfix(success=success, steps=steps, refresh=False)
            rollout_progress.close()

            successes += int(success)
            video_path = None
            if save_video and video_frames:
                video_path = args.output_dir / "videos" / f"task_{task_id:02d}_{task.name}" / f"episode_{episode_idx:03d}.mp4"
                write_rollout_video(video_frames, video_path)

            task_result["episodes"].append(
                {
                    "episode_index": episode_idx,
                    "success": success,
                    "steps": steps,
                    "video_path": str(video_path) if video_path is not None else None,
                }
            )
            task_result["success_rate"] = successes / float(episode_idx + 1)
            results["mean_success_rate"] = float(np.mean([t["success_rate"] for t in results["tasks"]]))
            write_results(args.output_dir, results)
            episode_progress.set_postfix(successes=successes, last_success=success, last_steps=steps, refresh=False)
        episode_progress.close()

        env.close()
        success_rate = successes / float(args.n_eval)
        task_result["success_rate"] = success_rate
        results["mean_success_rate"] = float(np.mean([t["success_rate"] for t in results["tasks"]]))
        write_results(args.output_dir, results)
        task_progress.set_postfix(task_id=task_id, success_rate=f"{success_rate:.3f}", task_name=task.name[:24], refresh=False)
    task_progress.close()

    results["mean_success_rate"] = float(np.mean([t["success_rate"] for t in results["tasks"]])) if results["tasks"] else 0.0
    write_results(args.output_dir, results)
    tqdm.write(f"Mean success rate: {results['mean_success_rate']:.4f}")


if __name__ == "__main__":
    main()
