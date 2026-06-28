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

"""No-cache DreamZero LIBERO server: the speed/accuracy reference.

Forces all 16 diffusion steps to run (``force_full_compute_env`` sets
``NUM_DIT_STEPS=16``), so it is a TRUE no-cache baseline. Note the stock
dreamzero_libero_eval server defaults to ``NUM_DIT_STEPS=8`` (i.e. an 8/16 static
cache!), so use THIS server for the reference number, not that one.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_common import CacheStats, force_full_compute_env, install_stats_only  # noqa: E402
from server_base import (  # noqa: E402
    add_common_model_args,
    build_model,
    resolve_device,
    serve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_full_compute_env()

    device = resolve_device(args)
    model, num_action_chunks = build_model(args, device)

    stats = CacheStats(method="none", out_path=args.stats_out)
    install_stats_only(model, stats)
    serve(model, num_action_chunks, device, args, cache_method="none", cache_config="full_compute_16steps")


if __name__ == "__main__":
    main()
