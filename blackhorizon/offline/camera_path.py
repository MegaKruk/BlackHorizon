"""Keyframed camera paths for offline rendering.

A path is a time-ordered list of orbital keyframes (distance,
inclination, azimuth, field of view around the hole at the origin);
frames between keyframes blend with smoothstep easing, giving
accelerate-decelerate motion without spline machinery. Azimuth is
interpolated on its unwrapped value so multi-revolution orbits are
expressed naturally (for example azimuth 0 to 720 for two turns).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..realtime.fly_camera import FlyCamera


@dataclass(frozen=True)
class CameraKeyframe:
    """One camera pose on the path timeline.

    Attributes:
        time: Timeline position in seconds.
        distance: Orbit radius in units of M.
        inclination_degrees: Polar angle from the spin axis.
        azimuth_degrees: Angle around the spin axis; not wrapped, so
            values beyond 360 describe additional revolutions.
        fov_degrees: Vertical field of view.
    """

    time: float
    distance: float
    inclination_degrees: float
    azimuth_degrees: float = 0.0
    fov_degrees: float = 70.0


def _smoothstep(u: float) -> float:
    """Cubic ease in [0, 1] with zero end slopes."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class CameraPath:
    """Time-parameterized camera motion built from keyframes."""

    def __init__(self, keyframes: list[CameraKeyframe]):
        """Validate ordering and store the keyframes.

        Args:
            keyframes: At least one keyframe, strictly increasing in
                time.
        """
        if not keyframes:
            raise ValueError("a camera path needs at least one keyframe")
        times = [k.time for k in keyframes]
        if any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
            raise ValueError("keyframe times must strictly increase")
        self._keyframes = list(keyframes)

    @property
    def duration(self) -> float:
        """Timeline length in seconds."""
        return self._keyframes[-1].time

    def pose_at(self, time: float) -> CameraKeyframe:
        """Interpolated pose at a timeline position (clamped to ends)."""
        frames = self._keyframes
        if time <= frames[0].time:
            return frames[0]
        if time >= frames[-1].time:
            return frames[-1]
        for k0, k1 in zip(frames, frames[1:]):
            if k0.time <= time <= k1.time:
                u = _smoothstep((time - k0.time) / (k1.time - k0.time))
                blend = lambda a, b: a + (b - a) * u
                return CameraKeyframe(
                    time=time,
                    distance=blend(k0.distance, k1.distance),
                    inclination_degrees=blend(
                        k0.inclination_degrees, k1.inclination_degrees
                    ),
                    azimuth_degrees=blend(
                        k0.azimuth_degrees, k1.azimuth_degrees
                    ),
                    fov_degrees=blend(k0.fov_degrees, k1.fov_degrees),
                )
        raise RuntimeError("unreachable: time inside validated range")

    def camera_at(self, time: float) -> FlyCamera:
        """FlyCamera looking at the hole for a timeline position."""
        pose = self.pose_at(time)
        return FlyCamera.from_orbit(
            pose.distance, pose.inclination_degrees, pose.azimuth_degrees
        )


def orbit_path(
    seconds: float,
    distance: float,
    inclination_degrees: float,
    revolutions: float = 1.0,
    fov_degrees: float = 70.0,
) -> CameraPath:
    """Convenience constructor: a constant-radius orbital sweep."""
    return CameraPath(
        [
            CameraKeyframe(
                0.0, distance, inclination_degrees, 0.0, fov_degrees
            ),
            CameraKeyframe(
                seconds,
                distance,
                inclination_degrees,
                360.0 * revolutions,
                fov_degrees,
            ),
        ]
    )
