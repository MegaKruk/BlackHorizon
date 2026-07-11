"""Shading traced rays into images.

Stage 1 shading is deliberately simple: captured rays are black (the
shadow), escaped rays sample a two-tone checkerboard on the celestial
sphere so gravitational lensing distortion is visible, and problematic
rays get diagnostic colors. Physically based disk emission arrives in
Stage 3 (see docs/DESIGN.md).
"""

from __future__ import annotations

import numpy
from PIL import Image

from .backend import to_numpy
from .geodesics import coordinate_velocity
from .kerr import KerrSpacetime
from .tracer import RayStatus, TraceResult

_COLOR_SHADOW = numpy.array([0, 0, 0], dtype=numpy.uint8)
_COLOR_CHECKER_DARK = numpy.array([24, 30, 52], dtype=numpy.uint8)
_COLOR_CHECKER_LIGHT = numpy.array([104, 128, 178], dtype=numpy.uint8)
_COLOR_MAX_STEPS = numpy.array([120, 32, 32], dtype=numpy.uint8)
_COLOR_FAILED = numpy.array([200, 0, 200], dtype=numpy.uint8)


def celestial_checkerboard(
    directions: numpy.ndarray, cells: int = 12
) -> numpy.ndarray:
    """Checkerboard colors for unit directions on the celestial sphere.

    Args:
        directions: Unit vectors, shape (n, 3).
        cells: Number of checker cells over the polar angle range.

    Returns:
        RGB colors, shape (n, 3), dtype uint8.
    """
    z = numpy.clip(directions[:, 2], -1.0, 1.0)
    theta = numpy.arccos(z)
    phi = numpy.arctan2(directions[:, 1], directions[:, 0])
    i_theta = numpy.floor(theta / numpy.pi * cells).astype(numpy.int64)
    i_phi = numpy.floor(
        (phi + numpy.pi) / (2.0 * numpy.pi) * 2 * cells
    ).astype(numpy.int64)
    parity = ((i_theta + i_phi) % 2).astype(bool)
    colors = numpy.where(
        parity[:, None], _COLOR_CHECKER_LIGHT, _COLOR_CHECKER_DARK
    )
    return colors.astype(numpy.uint8)


def render_image(
    spacetime: KerrSpacetime,
    result: TraceResult,
    width: int,
    height: int,
) -> numpy.ndarray:
    """Convert a TraceResult into an RGB image array.

    Args:
        spacetime: Spacetime the rays were traced in (needed to evaluate
            final propagation directions).
        result: Output of tracer.trace_rays for height * width rays in
            row-major pixel order.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Image array of shape (height, width, 3), dtype uint8.
    """
    n = width * height
    status = to_numpy(result.status)
    if status.shape[0] != n:
        raise ValueError("result size does not match image dimensions")

    velocity = to_numpy(coordinate_velocity(spacetime, result.states))
    spatial = velocity[:, 1:4]
    norms = numpy.linalg.norm(spatial, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions = spatial / norms

    pixels = numpy.empty((n, 3), dtype=numpy.uint8)
    pixels[:] = _COLOR_FAILED
    captured = status == int(RayStatus.CAPTURED)
    escaped = status == int(RayStatus.ESCAPED)
    exhausted = status == int(RayStatus.MAX_STEPS)
    pixels[captured] = _COLOR_SHADOW
    pixels[escaped] = celestial_checkerboard(directions[escaped])
    pixels[exhausted] = _COLOR_MAX_STEPS
    return pixels.reshape(height, width, 3)


def save_png(image: numpy.ndarray, path: str) -> None:
    """Write an RGB uint8 image array of shape (h, w, 3) to a PNG file."""
    Image.fromarray(image, mode="RGB").save(path)
