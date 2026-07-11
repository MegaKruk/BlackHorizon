"""Array backend dispatch.

All physics code in Black Horizon is written against the array API shared by
NumPy and CuPy. This module is the single place that knows about both, so
every other module stays backend-agnostic: it simply asks which module a
given array belongs to and keeps computing with that module.
"""

from __future__ import annotations

from typing import Any

import numpy

try:
    import cupy as _cupy
except ImportError:
    _cupy = None

Array = Any
"""Type alias for a NumPy or CuPy ndarray."""


def gpu_available() -> bool:
    """Return True if CuPy is installed and a CUDA device is usable."""
    if _cupy is None:
        return False
    try:
        return _cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_xp(backend: str):
    """Return the array module for a backend name ('cpu' or 'gpu')."""
    if backend == "cpu":
        return numpy
    if backend == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "GPU backend requested but CuPy or a CUDA device is not "
                "available. Install with: pip install cupy-cuda12x"
            )
        return _cupy
    raise ValueError(f"Unknown backend {backend!r}, expected 'cpu' or 'gpu'.")


def xp_of(array: Array):
    """Return the array module (numpy or cupy) that owns the given array."""
    if _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy
    return numpy


def to_numpy(array: Array) -> numpy.ndarray:
    """Copy an array to host memory as a NumPy ndarray."""
    if _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy.asnumpy(array)
    return numpy.asarray(array)
