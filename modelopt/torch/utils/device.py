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

"""Backend-neutral PyTorch device helpers."""

from typing import Any, TypeAlias

import torch

DeviceLike: TypeAlias = torch.device | str | int | None

__all__ = [
    "DeviceLike",
    "accelerator_empty_cache",
    "accelerator_synchronize",
    "get_accelerator_device_count",
    "get_accelerator_memory_info",
    "get_accelerator_memory_stats",
    "get_accelerator_module",
    "is_accelerator_device",
    "reset_accelerator_peak_memory_stats",
    "resolve_device",
    "set_accelerator_device",
    "with_device_index",
]


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    """Resolve a device selection and validate that its backend is available.

    ``"auto"`` selects PyTorch's current accelerator and falls back to CPU when
    no accelerator is available. Integer values select an index on the current
    accelerator.
    """
    if device is None or device == "auto":
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        return accelerator if accelerator is not None else torch.device("cpu")

    if isinstance(device, int):
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is None:
            raise RuntimeError(f"Cannot select device index {device}: no accelerator is available.")
        device = torch.device(accelerator.type, device)

    resolved = torch.device(device)
    if resolved.type in {"cpu", "meta"}:
        return resolved

    try:
        module = torch.get_device_module(resolved)
    except RuntimeError as error:
        raise ValueError(f"Unsupported PyTorch device backend: {resolved.type!r}.") from error

    is_available = getattr(module, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(f"Requested device backend {resolved.type!r} is not available.")
    return resolved


def get_accelerator_module(device: DeviceLike = "auto") -> Any | None:
    """Return the PyTorch backend module for an accelerator, or ``None`` for CPU/meta."""
    resolved = resolve_device(device)
    if not is_accelerator_device(resolved):
        return None
    return torch.get_device_module(resolved)


def is_accelerator_device(device: DeviceLike = "auto") -> bool:
    """Return whether ``device`` is an available non-CPU accelerator."""
    return resolve_device(device).type not in {"cpu", "meta"}


def with_device_index(device: DeviceLike, index: int) -> torch.device:
    """Return ``device`` at ``index`` while leaving CPU/meta devices unchanged."""
    resolved = resolve_device(device)
    if not is_accelerator_device(resolved):
        return resolved
    return torch.device(resolved.type, index)


def set_accelerator_device(device: DeviceLike = "auto") -> torch.device:
    """Set the current accelerator device and return the resolved device."""
    resolved = resolve_device(device)
    if is_accelerator_device(resolved):
        if resolved.index is None:
            resolved = torch.device(resolved.type, torch.accelerator.current_device_index())
        torch.accelerator.set_device_index(resolved)
    return resolved


def get_accelerator_device_count(device: DeviceLike = "auto") -> int:
    """Return the number of devices for the selected accelerator backend."""
    module = get_accelerator_module(device)
    if module is None:
        return 0
    device_count = getattr(module, "device_count", None)
    return int(device_count()) if callable(device_count) else 1


def accelerator_empty_cache(device: DeviceLike = "auto") -> None:
    """Release unused cached memory for the selected accelerator, when supported."""
    module = get_accelerator_module(device)
    empty_cache = getattr(module, "empty_cache", None) if module is not None else None
    if callable(empty_cache):
        empty_cache()


def accelerator_synchronize(device: DeviceLike = "auto") -> None:
    """Wait for queued work on the selected accelerator, when applicable."""
    resolved = resolve_device(device)
    if is_accelerator_device(resolved):
        torch.accelerator.synchronize(resolved)


def get_accelerator_memory_info(device: DeviceLike = "auto") -> tuple[int, int] | None:
    """Return ``(free, total)`` bytes for an accelerator, or ``None`` if unsupported."""
    resolved = resolve_device(device)
    module = get_accelerator_module(resolved)
    mem_get_info = getattr(module, "mem_get_info", None) if module is not None else None
    if not callable(mem_get_info):
        return None
    try:
        free, total = mem_get_info(resolved)
    except TypeError:
        free, total = mem_get_info()
    except (NotImplementedError, RuntimeError):
        return None
    return int(free), int(total)


def _memory_stat(module: Any | None, name: str, device: torch.device) -> int:
    method = getattr(module, name, None) if module is not None else None
    if not callable(method):
        return 0
    try:
        return int(method(device))
    except TypeError:
        return int(method())
    except (NotImplementedError, RuntimeError):
        return 0


def get_accelerator_memory_stats(device: DeviceLike = "auto") -> dict[str, int]:
    """Return allocator statistics in bytes for the selected accelerator."""
    resolved = resolve_device(device)
    module = get_accelerator_module(resolved)
    return {
        "allocated": _memory_stat(module, "memory_allocated", resolved),
        "max_allocated": _memory_stat(module, "max_memory_allocated", resolved),
        "reserved": _memory_stat(module, "memory_reserved", resolved),
        "max_reserved": _memory_stat(module, "max_memory_reserved", resolved),
    }


def reset_accelerator_peak_memory_stats(device: DeviceLike = "auto") -> None:
    """Reset peak allocator statistics for the selected accelerator, when supported."""
    resolved = resolve_device(device)
    module = get_accelerator_module(resolved)
    reset = getattr(module, "reset_peak_memory_stats", None) if module is not None else None
    if callable(reset):
        try:
            reset(resolved)
        except TypeError:
            reset()
