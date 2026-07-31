# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from modelopt.torch.utils import perf


def test_timer_uses_backend_neutral_clock_and_sync(monkeypatch):
    timestamps = iter([2.0, 2.025])
    synchronized = []
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(perf, "accelerator_synchronize", lambda device: synchronized.append(device))

    timer = perf.Timer(device="cpu")

    assert timer.stop() == pytest.approx(25.0)
    assert synchronized == [torch.device("cpu"), torch.device("cpu")]


def test_accumulating_timer_runs_on_cpu(monkeypatch):
    timestamps = iter([4.0, 4.01])
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(perf, "accelerator_synchronize", lambda device: None)
    perf.AccumulatingTimer.reset()

    with perf.AccumulatingTimer("cpu-work", device="cpu"):
        pass

    assert perf.AccumulatingTimer.get_total_time("cpu-work") == pytest.approx(10.0)
    assert perf.AccumulatingTimer.get_call_count("cpu-work") == 1


@pytest.mark.parametrize(
    ("memory_info", "expected"),
    [((25, 100), 0.75), ((0, 0), 0.0), (None, 0.0)],
)
def test_used_memory_fraction_handles_backend_reporting(monkeypatch, memory_info, expected):
    monkeypatch.setattr(perf, "get_accelerator_memory_info", lambda device: memory_info)

    assert perf.get_used_gpu_mem_fraction("cpu") == expected


def test_cuda_compatibility_aliases_use_generic_helpers(monkeypatch):
    calls = []
    stats = {"allocated": 1, "max_allocated": 2, "reserved": 3, "max_reserved": 4}
    monkeypatch.setattr(perf, "accelerator_empty_cache", lambda: calls.append("clear"))
    monkeypatch.setattr(perf, "get_accelerator_memory_stats", lambda device: stats)

    perf.clear_cuda_cache()

    assert perf.get_cuda_memory_stats("xpu:0") == stats
    assert calls == ["clear"]
