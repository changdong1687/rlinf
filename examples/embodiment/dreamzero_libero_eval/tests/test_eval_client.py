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

"""Offline unit tests for DreamZeroLiberoClient chunk-execution logic.

These exercise the open-loop / re-query / action-passthrough behavior against a fake
in-memory transport, so they run with only numpy installed (no GPU, no simulator, no
websockets). Run with:

    python -m pytest examples/embodiment/dreamzero_libero_eval/tests/test_eval_client.py
    # or simply:
    python examples/embodiment/dreamzero_libero_eval/tests/test_eval_client.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libero_eval_client import DreamZeroLiberoClient  # noqa: E402

CHUNK_LEN = 16
ACTION_DIM = 7


def make_chunk() -> np.ndarray:
    """Distinct, identifiable rows: row i = [i, i, ..., i]."""
    return np.tile(np.arange(CHUNK_LEN, dtype=np.float32)[:, None], (1, ACTION_DIM))


class FakeTransport:
    """In-memory stand-in for PickleWebsocketClient. Returns a fixed chunk, records calls."""

    def __init__(self, chunk: np.ndarray):
        self._chunk = chunk
        self.metadata = {"embodiment": "libero_sim", "action_dim": ACTION_DIM, "num_action_chunks": CHUNK_LEN}
        self.infer_requests = []
        self.reset_calls = []

    def infer(self, obs: dict) -> dict:
        self.infer_requests.append(obs)
        return {"actions": self._chunk.copy()}

    def reset(self, reset_info: dict | None = None) -> None:
        self.reset_calls.append(reset_info)


def _dummy_obs() -> dict:
    return {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


def _run_steps(client: DreamZeroLiberoClient, n: int) -> list[np.ndarray]:
    obs = _dummy_obs()
    return [client.infer(obs, "pick up the thing") for _ in range(n)]


def test_full_chunk_requeries_every_chunk_len():
    transport = FakeTransport(make_chunk())
    client = DreamZeroLiberoClient(host="x", port=0, open_loop_horizon=None, transport=transport)
    actions = _run_steps(client, 2 * CHUNK_LEN)
    # 32 env steps with full-chunk execution -> exactly 2 server requests.
    assert len(transport.infer_requests) == 2, len(transport.infer_requests)
    # Actions cycle through the chunk rows in order.
    for i, a in enumerate(actions):
        assert np.allclose(a, float(i % CHUNK_LEN)), (i, a[0])


def test_open_loop_horizon_8_requeries_twice_as_often():
    transport = FakeTransport(make_chunk())
    client = DreamZeroLiberoClient(host="x", port=0, open_loop_horizon=8, transport=transport)
    actions = _run_steps(client, 2 * CHUNK_LEN)
    # horizon 8 over 32 steps -> 4 requests.
    assert len(transport.infer_requests) == 4, len(transport.infer_requests)
    # Each request restarts the chunk at row 0; horizon stops at row 8 then re-queries.
    expected = [i % 8 for i in range(2 * CHUNK_LEN)]
    for a, e in zip(actions, expected):
        assert np.allclose(a, float(e)), (a[0], e)


def test_negative_horizon_is_treated_as_full_chunk():
    # CLI passes -1 -> main() converts to None; guard the class-level fallback too.
    transport = FakeTransport(make_chunk())
    client = DreamZeroLiberoClient(host="x", port=0, open_loop_horizon=-1, transport=transport)
    _run_steps(client, CHUNK_LEN + 1)
    # 17 steps: first 16 from one chunk, 17th forces a second request.
    assert len(transport.infer_requests) == 2, len(transport.infer_requests)


def test_reset_forces_new_request_and_new_session():
    transport = FakeTransport(make_chunk())
    client = DreamZeroLiberoClient(host="x", port=0, open_loop_horizon=None, transport=transport)
    sid0 = client.session_id
    _run_steps(client, 3)  # 1 request, chunk cached
    assert len(transport.infer_requests) == 1
    client.reset()
    assert len(transport.reset_calls) == 1
    assert transport.reset_calls[0]["session_id"] == sid0  # reset sent with the OLD session id
    assert client.session_id != sid0  # client rotates to a fresh session
    _run_steps(client, 1)  # must re-query after reset
    assert len(transport.infer_requests) == 2


def test_request_contains_required_keys():
    transport = FakeTransport(make_chunk())
    client = DreamZeroLiberoClient(host="x", port=0, transport=transport)
    _run_steps(client, 1)
    req = transport.infer_requests[0]
    for key in (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "prompt",
        "session_id",
    ):
        assert key in req, key
    assert req["prompt"] == "pick up the thing"


def test_action_shape_validation_rejects_bad_chunk():
    bad = FakeTransport(np.zeros((CHUNK_LEN, 8), dtype=np.float32))  # wrong action dim
    client = DreamZeroLiberoClient(host="x", port=0, transport=bad)
    try:
        _run_steps(client, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for action chunk with shape (N, 8)")


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
