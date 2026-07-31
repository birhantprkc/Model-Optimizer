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

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.utils import device as device_utils


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda check_available: None)

    assert device_utils.resolve_device("auto") == torch.device("cpu")


def test_resolve_device_auto_uses_current_accelerator(monkeypatch):
    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda check_available: torch.device("xpu"),
    )

    assert device_utils.resolve_device("auto") == torch.device("xpu")


def test_resolve_device_rejects_unavailable_backend(monkeypatch):
    backend = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(torch, "get_device_module", lambda device: backend)

    with pytest.raises(RuntimeError, match="not available"):
        device_utils.resolve_device("cuda")


def test_accelerator_helpers_use_selected_backend(monkeypatch):
    calls = []
    backend = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 3,
        empty_cache=lambda: calls.append("empty_cache"),
        mem_get_info=lambda device: (75, 100),
        memory_allocated=lambda device: 10,
        max_memory_allocated=lambda device: 20,
        memory_reserved=lambda device: 30,
        max_memory_reserved=lambda device: 40,
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: backend)
    monkeypatch.setattr(
        torch.accelerator, "set_device_index", lambda device: calls.append(("set", device))
    )
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda device: calls.append(("sync", device))
    )

    device = torch.device("xpu:1")
    assert device_utils.get_accelerator_device_count(device) == 3
    assert device_utils.get_accelerator_memory_info(device) == (75, 100)
    assert device_utils.get_accelerator_memory_stats(device) == {
        "allocated": 10,
        "max_allocated": 20,
        "reserved": 30,
        "max_reserved": 40,
    }
    device_utils.set_accelerator_device(device)
    device_utils.accelerator_synchronize(device)
    device_utils.accelerator_empty_cache(device)
    device_utils.reset_accelerator_peak_memory_stats(device)

    assert calls == [
        ("set", device),
        ("sync", device),
        "empty_cache",
        ("reset", device),
    ]


def test_set_accelerator_device_adds_current_index(monkeypatch):
    selected = []
    backend = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch, "get_device_module", lambda device: backend)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    monkeypatch.setattr(torch.accelerator, "set_device_index", selected.append)

    resolved = device_utils.set_accelerator_device("xpu")

    assert resolved == torch.device("xpu:2")
    assert selected == [torch.device("xpu:2")]


def test_cpu_helpers_are_safe_noops():
    assert device_utils.get_accelerator_device_count("cpu") == 0
    assert device_utils.get_accelerator_memory_info("cpu") is None
    assert device_utils.get_accelerator_memory_stats("cpu") == {
        "allocated": 0,
        "max_allocated": 0,
        "reserved": 0,
        "max_reserved": 0,
    }
    device_utils.set_accelerator_device("cpu")
    device_utils.accelerator_synchronize("cpu")
    device_utils.accelerator_empty_cache("cpu")
    device_utils.reset_accelerator_peak_memory_stats("cpu")


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU is not available"
)
def test_xpu_helpers_smoke():
    device = device_utils.resolve_device("xpu:0")
    tensor = torch.ones(1, device=device)

    assert tensor.device.type == "xpu"
    assert device_utils.get_accelerator_device_count(device) > 0
    assert device_utils.get_accelerator_memory_info(device) is not None
    device_utils.accelerator_synchronize(device)
