"""Batched geodesic ray tracing with adaptive stepping and termination.

The tracer propagates a batch of geodesics until each one either crosses
the event horizon (captured), leaves the scene (escaped), exhausts its step
budget, or fails numerically. Finished rays are frozen and removed from the
active set, so late iterations only pay for the few rays still orbiting
near the photon shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .backend import Array, xp_of
from .geodesics import geodesic_rhs
from .integrators import dormand_prince_step, error_ratio, step_factor
from .kerr import KerrSpacetime


class RayStatus(IntEnum):
    """Terminal state of a traced geodesic."""

    IN_FLIGHT = 0
    CAPTURED = 1
    ESCAPED = 2
    MAX_STEPS = 3
    FAILED = 4
    DISK = 5


@dataclass(frozen=True)
class TraceResult:
    """Outcome of tracing a batch of geodesics.

    Attributes:
        states: Final states, shape (n, 8).
        status: Per-ray RayStatus value, shape (n,), integer dtype.
        steps: Accepted integration steps per ray, shape (n,).
        iterations: Total controller iterations executed for the batch.
    """

    states: Array
    status: Array
    steps: Array
    iterations: int


def trace_rays(
    spacetime: KerrSpacetime,
    state0: Array,
    escape_radius: float,
    max_steps: int = 20000,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    initial_step: float = 0.1,
    max_step: float = 2.0,
    min_step: float = 1e-10,
    capture_margin: float = 1e-3,
) -> TraceResult:
    """Trace a batch of geodesics to termination.

    Args:
        spacetime: The Kerr spacetime to trace in.
        state0: Initial states, shape (n, 8), as built by
            geodesics.build_state.
        escape_radius: Kerr-Schild radius beyond which a ray counts as
            escaped. Must exceed the largest initial radius.
        max_steps: Maximum accepted steps per ray before giving up.
        rtol: Relative tolerance of the adaptive controller.
        atol: Absolute tolerance of the adaptive controller.
        initial_step: Initial affine-parameter step size.
        max_step: Upper bound on the step size (limits overshoot through
            the capture region).
        min_step: Below this step size a ray is marked FAILED.
        capture_margin: Rays are captured at r <= r_plus (1 + margin).

    Returns:
        A TraceResult with final states and per-ray diagnostics.
    """
    xp = xp_of(state0)
    n = state0.shape[0]
    capture_radius = spacetime.outer_horizon_radius * (1.0 + capture_margin)

    states = state0.copy()
    status = xp.full((n,), int(RayStatus.IN_FLIGHT), dtype=xp.int32)
    steps = xp.zeros((n,), dtype=xp.int64)
    h = xp.full((n,), float(initial_step), dtype=state0.dtype)

    def rhs(batch: Array) -> Array:
        return geodesic_rhs(spacetime, batch)

    # Rays created outside the scene bounds are malformed input.
    r0 = spacetime.kerr_schild_radius(
        state0[:, 1], state0[:, 2], state0[:, 3]
    )
    if bool(xp.any(r0 >= escape_radius)):
        raise ValueError("escape_radius must exceed all initial radii")

    max_iterations = 4 * max_steps
    iterations = 0
    while iterations < max_iterations:
        active_idx = xp.nonzero(status == int(RayStatus.IN_FLIGHT))[0]
        if active_idx.size == 0:
            break
        iterations += 1

        y = states[active_idx]
        h_a = h[active_idx]
        y_new, err = dormand_prince_step(rhs, y, h_a)
        ratio = error_ratio(y, y_new, err, rtol, atol)
        accept = ratio <= 1.0

        y = xp.where(accept[:, None], y_new, y)
        states[active_idx] = y
        steps_a = steps[active_idx] + accept.astype(steps.dtype)
        steps[active_idx] = steps_a
        h_a = xp.clip(h_a * step_factor(ratio), 0.0, max_step)
        h[active_idx] = h_a

        radius = spacetime.kerr_schild_radius(y[:, 1], y[:, 2], y[:, 3])
        finite = xp.all(xp.isfinite(y), axis=-1)
        status_a = status[active_idx]
        status_a = xp.where(
            ~finite, xp.int32(int(RayStatus.FAILED)), status_a
        )
        status_a = xp.where(
            finite & (radius <= capture_radius),
            xp.int32(int(RayStatus.CAPTURED)),
            status_a,
        )
        status_a = xp.where(
            finite & (radius >= escape_radius),
            xp.int32(int(RayStatus.ESCAPED)),
            status_a,
        )
        in_flight = status_a == int(RayStatus.IN_FLIGHT)
        status_a = xp.where(
            in_flight & (h_a < min_step),
            xp.int32(int(RayStatus.FAILED)),
            status_a,
        )
        status_a = xp.where(
            in_flight & (steps_a >= max_steps),
            xp.int32(int(RayStatus.MAX_STEPS)),
            status_a,
        )
        status[active_idx] = status_a

    still_active = status == int(RayStatus.IN_FLIGHT)
    status = xp.where(
        still_active, xp.int32(int(RayStatus.MAX_STEPS)), status
    )
    return TraceResult(
        states=states, status=status, steps=steps, iterations=iterations
    )
