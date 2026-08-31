# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pure-function tests for ``modelopt.torch.utils.distributed``."""

import pytest
import torch
import torch.nn as nn

from modelopt.torch.utils import distributed
from modelopt.torch.utils.distributed import _off_dtype_params


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


def _model(*sizes_and_dtypes) -> nn.Module:
    model = nn.Module()
    for i, (numel, dtype) in enumerate(sizes_and_dtypes):
        model.register_parameter(f"p{i}", nn.Parameter(torch.zeros(numel, dtype=dtype)))
    return model


def test_off_dtype_params_uniform_is_empty(recwarn):
    model = _model((8, torch.bfloat16), (4, torch.bfloat16))
    assert _off_dtype_params(model) == set()
    assert not recwarn.list


def test_off_dtype_params_returns_only_the_minority():
    # An fp32 MoE router gate next to bf16 weights, as Nemotron-3-Nano ships it.
    model = _model((100, torch.bfloat16), (5, torch.float32))
    with pytest.warns(UserWarning, match="mixed parameter dtypes"):
        off = _off_dtype_params(model)
    assert off == {model.p1}


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        # Dominant dtype is by element count, not parameter count: three small fp32 params
        # lose to one large bf16 param, and vice versa.
        ([(100, torch.bfloat16), (5, torch.float32), (5, torch.float32), (5, torch.float32)], "p0"),
        (
            [(100, torch.float32), (5, torch.bfloat16), (5, torch.bfloat16), (5, torch.bfloat16)],
            "p0",
        ),
    ],
)
def test_off_dtype_params_dominant_is_by_numel(params, expected):
    model = _model(*params)
    with pytest.warns(UserWarning, match="mixed parameter dtypes"):
        off = _off_dtype_params(model)
    kept = {n for n, p in model.named_parameters() if p not in off}
    assert kept == {expected}
