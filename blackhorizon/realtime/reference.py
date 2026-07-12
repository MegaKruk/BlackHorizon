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
    """Classification of rays traced with the shader algorithm."""

    status: numpy.ndarray
    positions: numpy.ndarray
    momenta: numpy.ndarray


def trace_like_shader(
    spacetime: KerrSpacetime,
    state0: numpy.ndarray,
    settings: RenderSettings,
    escape_radius: float,
) -> ReferenceResult:
    """Trace rays exactly as the fragment shader does.

    Fixed RK4 steps with h = clip(step_scale * (r - r_plus), min_step,
    max_step), terminating on capture, escape, or the step budget. Rays
    that exhaust the budget are reported as MAX_STEPS; the shader colors
    them black, which the tests account for.

    Args:
        spacetime: The Kerr spacetime.
        state0: Initial (n, 8) states from geodesics.build_state.
        settings: Real-time settings supplying the heuristic parameters.
        escape_radius: Radius at which rays count as escaped.

    Returns:
        A ReferenceResult with final status, positions, and momenta.
    """
    r_plus = spacetime.outer_horizon_radius
    capture_radius = r_plus * (1.0 + settings.capture_margin)
    n = state0.shape[0]
    state = state0.copy()
    status = numpy.full((n,), int(RayStatus.MAX_STEPS), dtype=numpy.int32)
    active = numpy.ones((n,), dtype=bool)

    for _ in range(settings.max_steps):
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
        escaped = ~captured & (radius >= escape_radius)
        status[idx[captured]] = int(RayStatus.CAPTURED)
        status[idx[escaped]] = int(RayStatus.ESCAPED)
        active[idx] = ~(captured | escaped)
        keep = ~(captured | escaped)
        idx = idx[keep]
        if idx.size == 0:
            break
        y = y[keep]
        radius = radius[keep]

        h = numpy.clip(
            settings.step_scale * (radius - r_plus),
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
        state[idx] = y + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ReferenceResult(
        status=status,
        positions=state[:, 1:4].copy(),
        momenta=state[:, 4:8].copy(),
    )
