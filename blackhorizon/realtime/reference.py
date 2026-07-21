"""Reference implementation of the real-time shader algorithm.

The fragment shader integrates with fixed-order RK4 and a distance-based
step heuristic instead of the embedded error control used by the Stage 1
tracer. This module mirrors that algorithm exactly (in float64 NumPy) so
its accuracy can be quantified against the validated adaptive tracer in
tests, and so future shader changes have a place to be checked before
they are transcribed to GLSL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from ..geodesics import geodesic_rhs
from ..kerr import KerrSpacetime
from ..tracer import RayStatus
from .settings import RenderSettings


@dataclass(frozen=True)
class ReferenceResult:
    """Classification of rays traced with the shader algorithm.

    When a disk is configured, rays terminating on it carry
    RayStatus.DISK, hit_radii holds their Kerr-Schild crossing radius,
    and hit_positions / hit_momenta the linearly interpolated crossing
    state, mirroring the shader exactly.
    """

    status: numpy.ndarray
    positions: numpy.ndarray
    momenta: numpy.ndarray
    hit_radii: numpy.ndarray | None = None
    hit_positions: numpy.ndarray | None = None
    hit_momenta: numpy.ndarray | None = None


def trace_like_shader(
    spacetime: KerrSpacetime,
    state0: numpy.ndarray,
    settings: RenderSettings,
    escape_radius: float,
    disk_radii: tuple[float, float] | None = None,
    interior_stop: float | None = None,
) -> ReferenceResult:
    """Trace rays exactly as the fragment shader does.

    Fixed RK4 steps with a piecewise heuristic: outside the horizon
    h = clip(step_scale * (r - r_plus), min_step, max_step), preserving
    the fine resolution that winding near-horizon rays need; inside,
    h = clip(step_scale * r / 2, ...), since no photon orbits exist
    there, Kerr-Schild is regular at the crossing, and steps must
    shrink toward the singularity. Terminates on capture, escape, or
    the step budget. Rays
    that exhaust the budget are reported as MAX_STEPS; the shader colors
    them black, which the tests account for.

    Args:
        spacetime: The Kerr spacetime.
        state0: Initial (n, 8) states from geodesics.build_state.
        settings: Real-time settings supplying the heuristic parameters.
        escape_radius: Radius at which rays count as escaped.
        disk_radii: Optional (inner, outer) disk radii; when given, an
            opaque equatorial disk terminates crossing rays exactly as
            the shader does.
        interior_stop: When set (interior camera mode), rays are not
            captured at the horizon; instead they terminate with
            RayStatus.TERMINATED at this small radius near the
            singularity. Rays crossing the horizon outward, backward in
            time, propagate normally, which is how an infalling camera
            still sees the outside universe.

    Returns:
        A ReferenceResult with final status, positions, and momenta.
    """
    r_plus = spacetime.outer_horizon_radius
    if interior_stop is None:
        capture_radius = r_plus * (1.0 + settings.capture_margin)
        capture_status = int(RayStatus.CAPTURED)
    else:
        capture_radius = interior_stop
        capture_status = int(RayStatus.TERMINATED)
    n = state0.shape[0]
    state = state0.copy()
    status = numpy.full((n,), int(RayStatus.MAX_STEPS), dtype=numpy.int32)
    active = numpy.ones((n,), dtype=bool)
    hit_radii = numpy.full((n,), numpy.nan)
    hit_positions = numpy.full((n, 3), numpy.nan)
    hit_momenta = numpy.full((n, 4), numpy.nan)

    max_steps = settings.max_steps
    if interior_stop is not None:
        # Mirror the shader's interior step-budget cap.
        max_steps = min(max_steps, 1024)
    for _ in range(max_steps):
        idx = numpy.nonzero(active)[0]
        if idx.size == 0:
            break
        y = state[idx]
        radius = spacetime.kerr_schild_radius(y[:, 1], y[:, 2], y[:, 3])
        momentum2 = numpy.sum(y[:, 5:8] * y[:, 5:8], axis=-1)

        # Past-directed rays inside the shadow asymptote to the horizon
        # with unbounded blueshift; a diverging momentum therefore counts
        # as capture (see RenderSettings.momentum_bailout).
        captured = (radius <= capture_radius) | (
            momentum2 >= settings.momentum_bailout**2
        )
        if interior_stop is not None:
            # Mirror the shader: rays hovering just above the terminal
            # surface with firmly growing momentum are on it already.
            captured = captured | (
                (radius <= capture_radius * 1.002)
                & (momentum2 > 2500.0)
            )
        escaped = ~captured & (radius >= escape_radius)
        status[idx[captured]] = capture_status
        status[idx[escaped]] = int(RayStatus.ESCAPED)
        active[idx] = ~(captured | escaped)
        keep = ~(captured | escaped)
        idx = idx[keep]
        if idx.size == 0:
            break
        y = y[keep]
        radius = radius[keep]

        if interior_stop is None:
            scale_length = radius - r_plus
        else:
            # Interior cameras: floor near the horizon (regular in
            # Kerr-Schild; rays linger while crossing), cap at r/2 so
            # steps shrink toward the singularity.
            scale_length = numpy.minimum(
                numpy.maximum(
                    numpy.abs(radius - r_plus), 0.15 * radius
                ),
                0.5 * radius,
            )
        h = numpy.clip(
            settings.step_scale * scale_length,
            settings.min_step,
            settings.max_step,
        )
        # Bound the displacement of any RK stage: blueshifting rays must
        # not be able to step across the interior in one evaluation.
        h = numpy.minimum(h, 1.0 / numpy.maximum(1.0, numpy.sqrt(momentum2[keep])))
        hh = h[:, None]
        k1 = geodesic_rhs(spacetime, y)
        k2 = geodesic_rhs(spacetime, y + 0.5 * hh * k1)
        k3 = geodesic_rhs(spacetime, y + 0.5 * hh * k2)
        k4 = geodesic_rhs(spacetime, y + hh * k3)
        y_new = y + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if disk_radii is not None:
            crossed = y[:, 3] * y_new[:, 3] < 0.0
            if bool(crossed.any()):
                t_cross = y[crossed, 3] / (
                    y[crossed, 3] - y_new[crossed, 3]
                )
                interpolated = (
                    y[crossed] + t_cross[:, None] * (y_new[crossed] - y[crossed])
                )
                r_hit = spacetime.kerr_schild_radius(
                    interpolated[:, 1], interpolated[:, 2], interpolated[:, 3]
                )
                on_disk = (r_hit >= disk_radii[0]) & (r_hit <= disk_radii[1])
                hit_local = numpy.zeros(y.shape[0], dtype=bool)
                hit_local[numpy.nonzero(crossed)[0][on_disk]] = True
                targets = idx[hit_local]
                status[targets] = int(RayStatus.DISK)
                active[targets] = False
                hit_radii[targets] = r_hit[on_disk]
                hit_positions[targets] = interpolated[on_disk][:, 1:4]
                hit_momenta[targets] = interpolated[on_disk][:, 4:8]

        state[idx] = y_new

    return ReferenceResult(
        status=status,
        positions=state[:, 1:4].copy(),
        momenta=state[:, 4:8].copy(),
        hit_radii=hit_radii if disk_radii is not None else None,
        hit_positions=hit_positions if disk_radii is not None else None,
        hit_momenta=hit_momenta if disk_radii is not None else None,
    )
