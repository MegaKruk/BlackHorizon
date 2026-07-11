"""Pinhole camera model.

Single responsibility: map the pixel grid of an image to world-space ray
directions and a camera position. The camera knows nothing about spacetime;
initial photon momenta are built from its directions by the geodesics
module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy


def _normalize(v: numpy.ndarray) -> numpy.ndarray:
    norm = numpy.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("zero-length vector cannot be normalized")
    return v / norm


@dataclass(frozen=True)
class PinholeCamera:
    """A pinhole camera looking at a target point.

    Attributes:
        position: Camera location in Cartesian Kerr-Schild coordinates.
        target: Point the camera looks at (default: the black hole).
        up_hint: Approximate up direction used to build the camera basis.
        fov_degrees: Horizontal field of view.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    position: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_degrees: float = 20.0
    width: int = 320
    height: int = 240

    forward: numpy.ndarray = field(init=False, repr=False)
    right: numpy.ndarray = field(init=False, repr=False)
    up: numpy.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 0.0 < self.fov_degrees < 180.0:
            raise ValueError("fov_degrees must be in (0, 180)")
        forward = _normalize(
            numpy.asarray(self.target, dtype=numpy.float64)
            - numpy.asarray(self.position, dtype=numpy.float64)
        )
        up_hint = numpy.asarray(self.up_hint, dtype=numpy.float64)
        right = numpy.cross(forward, up_hint)
        if numpy.linalg.norm(right) < 1e-12:
            # Looking along the up hint; fall back to a safe alternative.
            right = numpy.cross(forward, numpy.array([0.0, 1.0, 0.0]))
        right = _normalize(right)
        up = numpy.cross(right, forward)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "up", up)

    @classmethod
    def from_orbit(
        cls,
        distance: float,
        inclination_degrees: float,
        azimuth_degrees: float = 0.0,
        **kwargs,
    ) -> "PinholeCamera":
        """Place the camera on a sphere around the origin.

        Args:
            distance: Radial distance from the origin.
            inclination_degrees: Polar angle from the +z spin axis; 90 is
                edge-on with the equatorial plane.
            azimuth_degrees: Azimuthal angle around the spin axis.
            **kwargs: Forwarded to the PinholeCamera constructor.
        """
        if distance <= 0.0:
            raise ValueError("distance must be positive")
        inc = math.radians(inclination_degrees)
        azi = math.radians(azimuth_degrees)
        position = (
            distance * math.sin(inc) * math.cos(azi),
            distance * math.sin(inc) * math.sin(azi),
            distance * math.cos(inc),
        )
        return cls(position=position, **kwargs)

    def ray_directions(self) -> numpy.ndarray:
        """Unit ray directions for every pixel, shape (height * width, 3).

        Pixels are ordered row-major from the top-left corner; the ray for
        pixel (row, col) is at index row * width + col.
        """
        half_w = math.tan(math.radians(self.fov_degrees) / 2.0)
        half_h = half_w * self.height / self.width
        cols = (numpy.arange(self.width) + 0.5) / self.width * 2.0 - 1.0
        rows = 1.0 - (numpy.arange(self.height) + 0.5) / self.height * 2.0
        u, v = numpy.meshgrid(cols * half_w, rows * half_h)
        directions = (
            self.forward[None, None, :]
            + u[..., None] * self.right[None, None, :]
            + v[..., None] * self.up[None, None, :]
        )
        directions /= numpy.linalg.norm(directions, axis=-1, keepdims=True)
        return directions.reshape(-1, 3)

    def ray_origins(self) -> numpy.ndarray:
        """Camera position repeated per pixel, shape (height * width, 3)."""
        origin = numpy.asarray(self.position, dtype=numpy.float64)
        return numpy.broadcast_to(
            origin, (self.height * self.width, 3)
        ).copy()
