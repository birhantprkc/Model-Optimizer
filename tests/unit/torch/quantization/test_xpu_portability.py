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

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import tensor_quant

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU is not available"
)


def test_nvfp4_quantize_and_forward_on_xpu():
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 32),
    ).to(device="xpu:0", dtype=torch.bfloat16)
    inputs = torch.randn(4, 32, device="xpu:0", dtype=torch.bfloat16)

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda converted: converted(inputs))
    outputs = model(inputs)

    assert outputs.device.type == "xpu"
    assert torch.isfinite(outputs).all()


def test_int8_custom_op_runs_on_xpu():
    inputs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device="xpu:0")
    amax = torch.tensor(1.0, device="xpu:0")

    outputs = tensor_quant.quantize_op(inputs, amax, 8, 0, False, True)

    expected = torch.round(inputs * 127) / 127
    assert outputs.device.type == "xpu"
    assert torch.equal(outputs, expected)


def test_nvfp4_conv3d_uses_portable_path_on_xpu():
    model = torch.nn.Conv3d(2, 3, kernel_size=3, bias=False).eval().to("xpu:0")
    inputs = torch.randn(1, 2, 4, 4, 4, device="xpu:0")

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda converted: converted(inputs))
    outputs = model(inputs)

    assert outputs.shape == (1, 3, 2, 2, 2)
    assert outputs.device.type == "xpu"
    assert torch.isfinite(outputs).all()
