# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""This file contains utility functions used by the quantization backend."""

import functools

import torch


@functools.cache
def xpu_kernel_ops():
    """Return the torch op namespace of an installed XPU kernel provider, or ``None``.

    Importing ``vllm_xpu_kernels`` registers its SYCL kernels under ``torch.ops._xpu_C``.
    """
    if getattr(torch, "xpu", None) is None or not torch.xpu.is_available():
        return None
    try:
        import vllm_xpu_kernels  # noqa: F401
    except ImportError:
        return None
    return getattr(torch.ops, "_xpu_C", None)


def has_xpu_kernel(op_name: str) -> bool:
    """Return whether an XPU kernel provider exposes ``op_name``."""
    ops = xpu_kernel_ops()
    return ops is not None and hasattr(ops, op_name)


def fp8_compatible():
    """Check if the current device supports FP8."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if torch.version.hip is not None:
            # ROCm reports gfx architectures: FP8 needs MI300-class (gfx942) or newer.
            return capability >= (9, 4)
        return capability >= (8, 9)
    return has_xpu_kernel("fp8_gemm_w8a16")


def fp4_compatible():
    """Check if the current device supports FP4."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if torch.version.hip is not None:
            # ROCm reports gfx architectures: FP4 arrives with gfx950 (MI350-class).
            return capability >= (9, 5)
        return capability >= (10, 0)
    return has_xpu_kernel("nvfp4_gemm")


def quantizer_matches_default_cfg(module, default_cfg: dict) -> bool:
    """Return whether ``module``'s input/weight quantizers match a default quant config."""
    # Import here to avoid a circular import at package init.
    import modelopt.torch.quantization as mtq

    if not hasattr(module, "input_quantizer") or not hasattr(module, "weight_quantizer"):
        return False

    quant_cfg_list: list = default_cfg["quant_cfg"]
    input_cfg = mtq.config.find_quant_cfg_entry_by_path(quant_cfg_list, "*input_quantizer").get(
        "cfg", {}
    )
    weight_cfg = mtq.config.find_quant_cfg_entry_by_path(quant_cfg_list, "*weight_quantizer").get(
        "cfg", {}
    )
    # cfg may be a list (SequentialQuantizer); fall back to the first element.
    if isinstance(input_cfg, list):
        input_cfg = input_cfg[0]
    if isinstance(weight_cfg, list):
        weight_cfg = weight_cfg[0]
    if not isinstance(input_cfg, dict) or not isinstance(weight_cfg, dict):
        return False

    for quantizer, cfg in (
        (module.input_quantizer, input_cfg),
        (module.weight_quantizer, weight_cfg),
    ):
        for key, value in cfg.items():
            # "enable" and "effective_bits" are config metadata without quantizer attributes.
            if key in ("enable", "effective_bits"):
                continue
            if not hasattr(quantizer, key) or getattr(quantizer, key) != value:
                return False
    return True
