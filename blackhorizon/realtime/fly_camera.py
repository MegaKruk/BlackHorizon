"""Free-flight camera for the interactive mode.

Pure math, no OpenGL or windowing dependencies: position plus yaw/pitch
orientation, local-frame movement, and an orthonormal basis for the
shader. The app layer feeds it input deltas; tests exercise it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy

_WORLD_UP = numpy.array([0.0, 0.0, 1.0])
_PITCH_LIMIT = math.radians(89.0)


@dataclass
class FlyCamera:
    """A yaw/pitch camera flying through Kerr-Schild coordinate space.

    Yaw is the azimuth of the view direction around the +z spin axis;
    pitch is its elevation above the x-y plane. Both are stored in
    radians. move_speed is in units of M per second.
    """

    position: numpy.ndarray = field(
        default_factory=lambda: numpy.array([30.0, 0.0, 3.0])
    )
    yaw: float = math.pi
    pitch: float = 0.0
    move_speed: float = 8.0
    mouse_sensitivity: float = 0.003

    @classmethod
    def from_orbit(
        cls, distance: float, inclination_degrees: float, **kwargs
    ) -> "FlyCamera":
        """Place the camera on a sphere around the origin, looking at it."""
        inc = math.radians(inclination_degrees)
        position = numpy.array(
            [distance * math.sin(inc), 0.0, distance * math.cos(inc)]
        )
        camera = cls(position=position, **kwargs)
        camera.look_at(numpy.zeros(3))
        return camera

    @property
    def distance_from_origin(self) -> float:
        """Distance from the black hole at the origin."""
        return float(numpy.linalg.norm(self.position))

    def basis(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Orthonormal (forward, right, up) vectors of the view frame."""
        cp = math.cos(self.pitch)
        forward = numpy.array(
            [
                cp * math.cos(self.yaw),
                cp * math.sin(self.yaw),
                math.sin(self.pitch),
            ]
        )
        right = numpy.cross(forward, _WORLD_UP)
        norm = numpy.linalg.norm(right)
        if norm < 1e-9:
            right = numpy.array([math.sin(self.yaw), -math.cos(self.yaw), 0.0])
        else:
            right = right / norm
        up = numpy.cross(right, forward)
        return forward, right, up

    def look_at(self, target: numpy.ndarray) -> None:
        """Point the camera at a world-space target."""
        offset = numpy.asarray(target, dtype=float) - self.position
        norm = numpy.linalg.norm(offset)
        if norm < 1e-12:
            return
        direction = offset / norm
        self.yaw = math.atan2(direction[1], direction[0])
        self.pitch = max(
            -_PITCH_LIMIT,
            min(_PITCH_LIMIT, math.asin(float(direction[2]))),
        )

    def rotate(self, delta_x_pixels: float, delta_y_pixels: float) -> None:
        """Apply a mouse-look rotation from pixel deltas.

        Positive delta_x looks right; positive delta_y (screen-down)
        looks down. Pitch is clamped away from the poles.
        """
        self.yaw -= delta_x_pixels * self.mouse_sensitivity
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.pitch -= delta_y_pixels * self.mouse_sensitivity
        self.pitch = max(-_PITCH_LIMIT, min(_PITCH_LIMIT, self.pitch))

    def move(self, local_direction: numpy.ndarray, dt: float, boost: float = 1.0) -> None:
        """Translate along the local (forward, right, up) frame.

        Args:
            local_direction: Components along (forward, right, up); it is
                normalized internally, so diagonals are not faster.
            dt: Frame time in seconds.
            boost: Speed multiplier (for example a sprint key).
        """
        norm = numpy.linalg.norm(local_direction)
        if norm < 1e-12 or dt <= 0.0:
            return
        forward, right, up = self.basis()
        world = (
            local_direction[0] * forward
            + local_direction[1] * right
            + local_direction[2] * up
        ) / norm
        self.position = self.position + world * self.move_speed * boost * dt
