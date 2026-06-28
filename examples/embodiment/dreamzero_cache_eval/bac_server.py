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

"""DreamZero LIBERO policy server with BAC (block-wise adaptive caching).

Requires a schedule JSON precomputed by analysis/bac_compute_schedule.py.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_common import CacheStats, force_full_compute_env  # noqa: E402
from server_base import (  # noqa: E402
    add_common_model_args,
    build_model,
    resolve_device,
    serve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_model_args(parser)
    parser.add_argument("--bac-schedule", type=str, required=True, help="Path to schedule.json from bac_compute_schedule.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_full_compute_env()

    device = resolve_device(args)
    model, num_action_chunks = build_model(args, device)

    from bac_dreamzero import install_bac

    stats = CacheStats(method="bac", out_path=args.stats_out)
    install_bac(model, schedule_path=args.bac_schedule, stats=stats)
    cfg = f"schedule={os.path.basename(args.bac_schedule)}"
    serve(model, num_action_chunks, device, args, cache_method="bac", cache_config=cfg)


if __name__ == "__main__":
    main()
