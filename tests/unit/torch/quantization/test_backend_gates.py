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
from modelopt.torch.quantization.backends import utils as backend_utils


def _fake_cuda(monkeypatch, capability, hip=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: capability)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)


@pytest.mark.parametrize(
    ("capability", "hip", "fp8", "fp4"),
    [
        ((8, 0), None, False, False),
        ((8, 9), None, True, False),
        ((10, 0), None, True, True),
        ((9, 0), "6.4.0", False, False),
        ((9, 4), "6.4.0", True, False),  # MI300-class
        ((9, 5), "6.4.0", True, True),  # MI350-class
    ],
)
def test_compatibility_gates_cuda_and_rocm(monkeypatch, capability, hip, fp8, fp4):
    _fake_cuda(monkeypatch, capability, hip)
    assert backend_utils.fp8_compatible() is fp8
    assert backend_utils.fp4_compatible() is fp4


def test_compatibility_gates_xpu_provider(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(backend_utils, "has_xpu_kernel", lambda op: True)
    assert backend_utils.fp8_compatible() is True
    assert backend_utils.fp4_compatible() is True

    monkeypatch.setattr(backend_utils, "has_xpu_kernel", lambda op: False)
    assert backend_utils.fp8_compatible() is False
    assert backend_utils.fp4_compatible() is False


def test_quantizer_matches_default_cfg():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    inputs = torch.randn(2, 32)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(inputs))
    module = model[0]

    assert backend_utils.quantizer_matches_default_cfg(module, mtq.NVFP4_DEFAULT_CFG)
    assert not backend_utils.quantizer_matches_default_cfg(module, mtq.FP8_DEFAULT_CFG)


def test_quantizer_matches_default_cfg_skips_config_only_keys():
    # Regression: NVFP4_DEFAULT_CFG carries "effective_bits", which TensorQuantizer does not
    # expose as an attribute; the config match must ignore such config-only metadata.
    weight_cfg = mtq.config.find_quant_cfg_entry_by_path(
        mtq.NVFP4_DEFAULT_CFG["quant_cfg"], "*weight_quantizer"
    ).get("cfg", {})
    if isinstance(weight_cfg, list):
        weight_cfg = weight_cfg[0]
    assert "effective_bits" in weight_cfg

    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    inputs = torch.randn(2, 32)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(inputs))
    assert not hasattr(model[0].input_quantizer, "effective_bits")
    assert backend_utils.quantizer_matches_default_cfg(model[0], mtq.NVFP4_DEFAULT_CFG)
