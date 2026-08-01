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

"""Real-quant GEMM implementations backed by external Intel XPU kernel libraries.

The ops come from the ``vllm_xpu_kernels`` package (``torch.ops._xpu_C``), which provides
ModelOpt-compatible NVFP4 W4A16 and FP8 W8A16 SYCL GEMMs. Activations are quantized through
the module's input quantizer (fake quant on the target grid), so results match the simulated
quantization numerics while the weight-side GEMM consumes packed data directly.
"""

import torch
from torch.autograd import Function

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.gemm_registry import gemm_registry
from modelopt.torch.quantization.backends.utils import quantizer_matches_default_cfg, xpu_kernel_ops
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor, QTensorWrapper
from modelopt.torch.quantization.utils import reduce_amax

__all__ = [
    "XpuFp8PerTensorLinear",
    "XpuNvfp4Linear",
    "xpu_fp8_per_tensor_gemm",
    "xpu_nvfp4_gemm",
]


def xpu_nvfp4_gemm(quant_module, input_tensor, bias=None):
    """NVFP4 W4A16 GEMM on XPU over a compressed weight; input follows the input quantizer."""
    ops = xpu_kernel_ops()
    weight = quant_module.weight.get_qtensor()

    input_tensor = quant_module.input_quantizer(input_tensor)
    input_shape = input_tensor.shape
    x = input_tensor.reshape(-1, input_shape[-1]).contiguous()

    output = ops.nvfp4_gemm(
        x,
        weight._quantized_data.contiguous(),
        quant_module.weight_quantizer._scale.contiguous(),
        float(quant_module.weight_quantizer._double_scale),
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], -1)


def xpu_fp8_per_tensor_gemm(quant_module, input_tensor, bias=None):
    """FP8 per-tensor GEMM on XPU (W8A16 kernel); input follows the input quantizer."""
    ops = xpu_kernel_ops()

    cached_scale_b = (
        hasattr(quant_module, "_scale_b") and quant_module.weight_quantizer.amax is not None
    )
    if not cached_scale_b:
        weight_amax = quant_module.weight_quantizer.amax
        if weight_amax is None:
            weight_amax = reduce_amax(quant_module.weight)
        assert weight_amax != 0
        quant_module._scale_b = (weight_amax.float() / 448.0).to(device=quant_module.weight.device)

    weight = quant_module.weight
    if weight.dtype != torch.float8_e4m3fn:
        weight_fp8 = (
            (weight.data / quant_module._scale_b).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        )
    else:
        weight_fp8 = weight.data

    input_dtype = input_tensor.dtype
    if input_dtype == torch.float32:
        input_tensor = input_tensor.to(torch.bfloat16)
    input_tensor = quant_module.input_quantizer(input_tensor)
    input_shape = input_tensor.shape
    x = input_tensor.reshape(-1, input_shape[-1]).contiguous()

    output = ops.fp8_gemm_w8a16(
        x,
        weight_fp8.transpose(0, 1),
        quant_module._scale_b.reshape(1),
        bias if bias is None else bias.to(x.dtype),
    )
    return output.reshape(*input_shape[:-1], -1).to(input_dtype)


class XpuNvfp4Linear(Function):
    """Linear layer with NVFP4 quantization on XPU."""

    @staticmethod
    def forward(
        ctx, quant_module, input_tensor, weight, bias=None, allreduce_dgrad=False, tp_group=None
    ):
        """Forward method."""
        ctx.save_for_backward(
            input_tensor if weight.requires_grad else None,
            weight if input_tensor.requires_grad else None,
            getattr(quant_module.weight_quantizer, "_scale", None),
            getattr(quant_module.weight_quantizer, "_double_scale", None),
        )
        ctx.compute_bias_grad = bias is not None and bias.requires_grad
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        return xpu_nvfp4_gemm(quant_module, input_tensor, bias)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward method using the portable dequantize path."""
        input_tensor, weight, scale, double_scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if weight is not None:
            if isinstance(weight, QTensorWrapper):
                weight = weight.get_qtensor()
                assert isinstance(weight, NVFP4QTensor)
                weight = weight.dequantize(
                    scale=scale, double_scale=double_scale, block_sizes={-1: 16}
                )
            grad_input = grad_outputs @ weight
        if input_tensor is not None:
            grad_weight = grad_outputs.reshape(-1, grad_outputs.shape[-1]).T @ input_tensor.reshape(
                -1, input_tensor.shape[-1]
            )
        if ctx.compute_bias_grad:
            grad_bias = grad_outputs.sum(dim=list(range(grad_outputs.dim() - 1)))
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return None, grad_input, grad_weight, grad_bias, None, None

    @classmethod
    def apply(cls, *args, **kwargs):
        """Get rid of kwargs because super does not support kwargs."""
        additional_args = tuple(kwargs.values())
        return super().apply(*args, *additional_args)


class XpuFp8PerTensorLinear(Function):
    """Linear layer with FP8 per tensor quantization on XPU."""

    @staticmethod
    def forward(
        ctx, quant_module, input_tensor, weight, bias=None, allreduce_dgrad=False, tp_group=None
    ):
        """Forward method."""
        ctx.save_for_backward(
            input_tensor if weight.requires_grad else None,
            weight if input_tensor.requires_grad else None,
            getattr(quant_module.weight_quantizer, "_scale", None),
        )
        ctx.compute_bias_grad = bias is not None and bias.requires_grad
        ctx.block_sizes = getattr(quant_module.weight_quantizer, "_block_sizes", None)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        return xpu_fp8_per_tensor_gemm(quant_module, input_tensor, bias)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward method using the portable dequantize path."""
        input_tensor, weight, scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if weight is not None:
            if isinstance(weight, QTensorWrapper):
                weight = weight.get_qtensor()
                assert isinstance(weight, FP8QTensor)
                weight = weight.dequantize(scale=scale, block_sizes=ctx.block_sizes)
            grad_input = grad_outputs @ weight
        if input_tensor is not None:
            grad_weight = grad_outputs.reshape(-1, grad_outputs.shape[-1]).T @ input_tensor.reshape(
                -1, input_tensor.shape[-1]
            )
        if ctx.compute_bias_grad:
            grad_bias = grad_outputs.sum(dim=list(range(grad_outputs.dim() - 1)))
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return None, grad_input, grad_weight, grad_bias, None, None

    @classmethod
    def apply(cls, *args, **kwargs):
        """Get rid of kwargs because super does not support kwargs."""
        additional_args = tuple(kwargs.values())
        return super().apply(*args, *additional_args)


def _xpu_common_checks(module, input, op_name):
    ops = xpu_kernel_ops()
    if ops is None or not hasattr(ops, op_name):
        return False
    if input.device.type != "xpu":
        return False

    # Import lazily because backend registration can run while quant_linear is initializing.
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear

    return isinstance(module, RealQuantLinear)


def _xpu_nvfp4_availability_check(module, input, args, kwargs):
    """Check whether the XPU NVFP4 W4A16 GEMM applies to this module and input."""
    if not _xpu_common_checks(module, input, "nvfp4_gemm"):
        return False
    if not quantizer_matches_default_cfg(module, mtq.NVFP4_DEFAULT_CFG):
        return False

    # The kernel consumes compressed weights in ModelOpt layout only.
    if not isinstance(module.weight, QTensorWrapper) or not isinstance(
        module.weight.get_qtensor(), NVFP4QTensor
    ):
        return False
    scale = getattr(module.weight_quantizer, "_scale", None)
    if scale is None or scale.dtype != torch.float8_e4m3fn or scale.dim() != 2:
        return False
    if getattr(module.weight_quantizer, "_double_scale", None) is None:
        return False

    # Kernel constraints: K must be a multiple of 32, block_size 16.
    if input.shape[-1] % 32 != 0 or scale.shape[-1] * 16 != input.shape[-1]:
        return False
    return (
        module.input_quantizer.block_sizes[-1] == 16
        and module.weight_quantizer.block_sizes[-1] == 16
    )


def _xpu_fp8_availability_check(module, input, args, kwargs):
    """Check whether the XPU FP8 W8A16 GEMM applies to this module and input."""
    if not _xpu_common_checks(module, input, "fp8_gemm_w8a16"):
        return False
    if not quantizer_matches_default_cfg(module, mtq.FP8_DEFAULT_CFG):
        return False
    return input.shape[-1] % 16 == 0


gemm_registry.register(
    gemm_func=XpuNvfp4Linear.apply,
    availability_check=_xpu_nvfp4_availability_check,
)
gemm_registry.register(
    gemm_func=XpuFp8PerTensorLinear.apply,
    availability_check=_xpu_fp8_availability_check,
)
