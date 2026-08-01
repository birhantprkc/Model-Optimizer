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
from modelopt.torch.quantization.backends import enable_real_quant_gemm
from modelopt.torch.quantization.backends.utils import has_xpu_kernel
from modelopt.torch.quantization.backends.xpu_gemm import XpuFp8PerTensorLinear, XpuNvfp4Linear

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU is not available"
)


def _quantize_and_compress(config):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(256, 64),
    ).to(device="xpu:0", dtype=torch.bfloat16)
    inputs = torch.randn(8, 32, 128, device="xpu:0", dtype=torch.bfloat16)

    mtq.quantize(model, config, lambda m: m(inputs))
    with torch.no_grad():
        reference = model(inputs)

    mtq.compress(model)
    enable_real_quant_gemm(model)
    with torch.no_grad():
        outputs = model(inputs)
    return model, outputs, reference


@pytest.mark.skipif(not has_xpu_kernel("nvfp4_gemm"), reason="No XPU NVFP4 kernel provider")
def test_nvfp4_real_quant_gemm_on_xpu():
    model, outputs, reference = _quantize_and_compress(mtq.NVFP4_DEFAULT_CFG)

    impls = [
        m._real_quant_gemm_impl for m in model.modules() if hasattr(m, "_real_quant_gemm_impl")
    ]
    assert impls and all(impl == XpuNvfp4Linear.apply for impl in impls)

    relative_error = (
        outputs.float() - reference.float()
    ).abs().max() / reference.float().abs().max()
    assert relative_error < 0.15


@pytest.mark.skipif(not has_xpu_kernel("fp8_gemm_w8a16"), reason="No XPU FP8 kernel provider")
def test_fp8_real_quant_gemm_on_xpu():
    model, outputs, reference = _quantize_and_compress(mtq.FP8_DEFAULT_CFG)

    impls = [
        m._real_quant_gemm_impl for m in model.modules() if hasattr(m, "_real_quant_gemm_impl")
    ]
    assert impls and all(impl == XpuFp8PerTensorLinear.apply for impl in impls)

    relative_error = (
        outputs.float() - reference.float()
    ).abs().max() / reference.float().abs().max()
    assert relative_error < 0.15


@pytest.mark.skipif(not has_xpu_kernel("nvfp4_gemm"), reason="No XPU NVFP4 kernel provider")
def test_nvfp4_gemm_falls_back_on_unsupported_k():
    # K=48 is not a multiple of 32: the XPU kernel must not match, and the module
    # falls back to the simulated path instead of crashing.
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(48, 64)).to(device="xpu:0", dtype=torch.bfloat16)
    inputs = torch.randn(4, 48, device="xpu:0", dtype=torch.bfloat16)

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(inputs))
    mtq.compress(model)
    enable_real_quant_gemm(model)
    with torch.no_grad(), pytest.warns(UserWarning, match="No real-quant GEMM found"):
        outputs = model(inputs)
    assert torch.isfinite(outputs).all()
