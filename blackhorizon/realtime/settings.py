"""Render settings and quality presets for the real-time mode.

Single responsibility: hold and validate every tunable of the real-time
tracer and translate quality presets into concrete numbers. The engine
consumes these values as shader uniforms; the UI mutates them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class BackgroundMode(int, Enum):
    """Celestial sphere styles understood by the fragment shader."""

    CHECKERBOARD = 0
    STARFIELD = 1


class QualityPreset(str, Enum):
    """Named fidelity levels trading step budget for frame rate."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


_PRESET_VALUES: dict[QualityPreset, dict] = {
    QualityPreset.LOW: {
        "max_steps": 256,
        "step_scale": 0.35,
        "resolution_scale": 0.5,
    },
    QualityPreset.MEDIUM: {
        "max_steps": 512,
        "step_scale": 0.22,
        "resolution_scale": 0.75,
    },
    QualityPreset.HIGH: {
        "max_steps": 1024,
        "step_scale": 0.14,
        "resolution_scale": 1.0,
    },
    QualityPreset.ULTRA: {
        "max_steps": 2048,
        "step_scale": 0.08,
        "resolution_scale": 1.0,
    },
}

MAX_STEPS_HARD_LIMIT = 4096
"""Compile-time loop bound baked into the shader; runtime max_steps must
stay at or below this value."""


@dataclass
class RenderSettings:
    """All tunables of the real-time Kerr tracer.

    Attributes:
        spin: Black hole spin a/M in [-1, 1].
        fov_degrees: Horizontal field of view of the camera.
        max_steps: Runtime integration step budget per ray.
        step_scale: Step heuristic gain; the step is
            step_scale * (r - r_plus) clamped to [min_step, max_step].
        min_step: Lower clamp of the step heuristic.
        max_step: Upper clamp of the step heuristic.
        resolution_scale: Internal render resolution as a fraction of the
            window resolution, in (0, 1].
        capture_margin: Rays are captured at r <= r_plus (1 + margin).
            The real-time default is looser than Stage 1 because
            past-directed rays asymptote to the horizon with diverging
            blueshift; see momentum_bailout.
        escape_radius: Radius at which rays sample the background. If not
            positive, the engine derives it from the camera distance.
        disk_enabled: Whether the Novikov-Thorne disk is rendered.
        disk_outer_radius: Outer disk edge in units of M. The engine
            clamps the effective value above the spin-dependent ISCO.
        disk_temperature: Peak effective temperature of the disk in
            Kelvin; sets the emitted color palette before redshift.
        disk_detail: Strength of the procedural brightness streaks in
            [0, 1]; purely cosmetic.
        exposure: Linear brightness multiplier before tone mapping.
        momentum_bailout: Spatial momentum magnitude above which a ray
            is classified as captured. Past-directed rays entering the
            shadow blueshift without bound as they asymptote to the
            horizon, so a diverging momentum identifies capture before
            fixed-step integration can become unstable.
        background: Celestial sphere style.
    """

    spin: float = 0.9
    fov_degrees: float = 60.0
    max_steps: int = 512
    step_scale: float = 0.22
    min_step: float = 1e-4
    max_step: float = 4.0
    resolution_scale: float = 0.75
    capture_margin: float = 2e-2
    escape_radius: float = 0.0
    momentum_bailout: float = 1e3
    background: BackgroundMode = BackgroundMode.CHECKERBOARD
    disk_enabled: bool = True
    disk_outer_radius: float = 18.0
    disk_temperature: float = 6500.0
    disk_detail: float = 1.0
    exposure: float = 1.0

    def validate(self) -> None:
        """Raise ValueError if any field is outside its legal range."""
        if not -1.0 <= self.spin <= 1.0:
            raise ValueError("spin must be in [-1, 1]")
        if not 1.0 <= self.fov_degrees <= 170.0:
            raise ValueError("fov_degrees must be in [1, 170]")
        if not 1 <= self.max_steps <= MAX_STEPS_HARD_LIMIT:
            raise ValueError(
                f"max_steps must be in [1, {MAX_STEPS_HARD_LIMIT}]"
            )
        if self.step_scale <= 0.0:
            raise ValueError("step_scale must be positive")
        if not 0.0 < self.min_step <= self.max_step:
            raise ValueError("require 0 < min_step <= max_step")
        if not 0.05 <= self.resolution_scale <= 1.0:
            raise ValueError("resolution_scale must be in [0.05, 1]")
        if self.capture_margin < 0.0:
            raise ValueError("capture_margin must be non-negative")
        if self.momentum_bailout <= 1.0:
            raise ValueError("momentum_bailout must exceed 1")
        if self.disk_outer_radius <= 1.0:
            raise ValueError("disk_outer_radius must exceed 1 M")
        if not 500.0 <= self.disk_temperature <= 40000.0:
            raise ValueError("disk_temperature must be in [500, 40000] K")
        if not 0.0 <= self.disk_detail <= 1.0:
            raise ValueError("disk_detail must be in [0, 1]")
        if self.exposure <= 0.0:
            raise ValueError("exposure must be positive")

    def apply_preset(self, preset: QualityPreset) -> "RenderSettings":
        """Return a copy of the settings with a quality preset applied."""
        return replace(self, **_PRESET_VALUES[preset])

    def effective_escape_radius(self, camera_distance: float) -> float:
        """Escape radius to use for a camera at the given distance."""
        if self.escape_radius > 0.0:
            return self.escape_radius
        return max(2.0 * camera_distance, 100.0)
