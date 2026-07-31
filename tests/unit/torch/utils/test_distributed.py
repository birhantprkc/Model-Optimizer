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

import torch

from modelopt.torch.utils import distributed


def test_collective_device_uses_cpu_for_gloo(monkeypatch):
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")

    assert distributed._collective_device() == torch.device("cpu")


def test_collective_device_uses_current_accelerator_for_xccl(monkeypatch):
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "xccl")
    monkeypatch.setattr(distributed, "resolve_device", lambda device: torch.device("xpu:1"))

    assert distributed._collective_device() == torch.device("xpu:1")


def test_setup_selects_cpu_backend(monkeypatch):
    calls = []
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed, "set_accelerator_device", lambda device: calls.append(("set", device))
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_default_backend_for_device",
        lambda device: calls.append(("backend", device)) or "gloo",
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend, timeout=None: calls.append(("init", backend, timeout)),
    )

    device = distributed.setup(device="cpu")

    assert device == torch.device("cpu")
    assert calls == [
        ("set", torch.device("cpu")),
        ("backend", torch.device("cpu")),
        ("init", "gloo", None),
    ]


def test_setup_selects_accelerator_backend(monkeypatch):
    calls = []
    selected_device = torch.device("xpu:2")
    monkeypatch.setattr(distributed, "local_rank", lambda: 2)
    monkeypatch.setattr(distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed,
        "with_device_index",
        lambda device, index: selected_device,
    )
    monkeypatch.setattr(
        distributed, "set_accelerator_device", lambda device: calls.append(("set", device))
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_default_backend_for_device",
        lambda device: calls.append(("backend", device)) or "xccl",
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend, timeout=None: calls.append(("init", backend, timeout)),
    )

    device = distributed.setup(device="xpu")

    assert device == selected_device
    assert calls == [
        ("set", selected_device),
        ("backend", selected_device),
        ("init", "xccl", None),
    ]


def test_allgather_serializes_on_cpu_for_gloo(monkeypatch):
    call_count = 0

    def fake_allgather(outputs, tensor, group=None):
        nonlocal call_count
        call_count += 1
        for output in outputs:
            output.copy_(tensor)

    monkeypatch.setattr(distributed, "size", lambda group=None: 2)
    monkeypatch.setattr(distributed, "collective_device", lambda group=None: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_allgather", fake_allgather)

    assert distributed.allgather({"portable": True}) == [
        {"portable": True},
        {"portable": True},
    ]
    assert call_count == 2
