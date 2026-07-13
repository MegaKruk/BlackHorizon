"""Batched ordinary differential equation integrators.

The steppers are generic: they integrate d(state)/d(lambda) = rhs(state) for
a batch of independent samples, state shape (n, d), with a per-sample step
size h of shape (n,). This lets every ray or particle in a batch advance
with its own adaptive step while the whole batch stays vectorized on the
CPU or the GPU.

Provided schemes (see docs/DESIGN.md section 2.4):
    - rk4_step: classical fixed-step Runge-Kutta 4, used for benchmarking.
    - dormand_prince_step: embedded Dormand-Prince 5(4) returning the fifth
      order solution and a per-sample error estimate for adaptive control.
"""

from __future__ import annotations

from typing import Callable

from .backend import Array, xp_of

Rhs = Callable[[Array], Array]

# Dormand-Prince 5(4) Butcher tableau (autonomous form, c nodes unused).
_A21 = 1.0 / 5.0
_A31, _A32 = 3.0 / 40.0, 9.0 / 40.0
_A41, _A42, _A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
_A51, _A52, _A53, _A54 = (
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
)
_A61, _A62, _A63, _A64, _A65 = (
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
)
_A71, _A73, _A74, _A75, _A76 = (
    35.0 / 384.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
)
# Fifth-order weights (b7 = 0) and error weights e = b5 - b4.
_B1, _B3, _B4, _B5, _B6 = _A71, _A73, _A74, _A75, _A76
_E1 = 71.0 / 57600.0
_E3 = -71.0 / 16695.0
_E4 = 71.0 / 1920.0
_E5 = -17253.0 / 339200.0
_E6 = 22.0 / 525.0
_E7 = -1.0 / 40.0


def rk4_step(rhs: Rhs, state: Array, h: Array) -> Array:
    """Advance a batch one classical RK4 step.

    Args:
        rhs: Right-hand side function, (n, d) to (n, d).
        state: Current states, shape (n, d).
        h: Per-sample step sizes, shape (n,).

    Returns:
        New states, shape (n, d).
    """
    hh = h[:, None]
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * hh * k1)
    k3 = rhs(state + 0.5 * hh * k2)
    k4 = rhs(state + hh * k3)
    return state + (hh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def dormand_prince_step(
    rhs: Rhs, state: Array, h: Array
) -> tuple[Array, Array]:
    """Advance a batch one embedded Dormand-Prince 5(4) step.

    Args:
        rhs: Right-hand side function, (n, d) to (n, d).
        state: Current states, shape (n, d).
        h: Per-sample step sizes, shape (n,).

    Returns:
        Tuple (new_state, error) where new_state is the fifth-order
        solution and error is the per-component embedded error estimate,
        both of shape (n, d).
    """
    hh = h[:, None]
    k1 = rhs(state)
    k2 = rhs(state + hh * (_A21 * k1))
    k3 = rhs(state + hh * (_A31 * k1 + _A32 * k2))
    k4 = rhs(state + hh * (_A41 * k1 + _A42 * k2 + _A43 * k3))
    k5 = rhs(state + hh * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
    k6 = rhs(
        state
        + hh * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5)
    )
    increment = _B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6
    new_state = state + hh * increment
    k7 = rhs(new_state)
    error = hh * (
        _E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7
    )
    return new_state, error


def error_ratio(
    state: Array, new_state: Array, error: Array, rtol: float, atol: float
) -> Array:
    """Scaled RMS error per sample; a value <= 1 means the step is accepted."""
    xp = xp_of(state)
    scale = atol + rtol * xp.maximum(xp.abs(state), xp.abs(new_state))
    ratio = xp.sqrt(xp.mean((error / scale) ** 2, axis=-1))
    return xp.where(xp.isfinite(ratio), ratio, xp.inf)


def step_factor(ratio: Array) -> Array:
    """Step-size multiplier from the error ratio (PI-free basic controller)."""
    xp = xp_of(ratio)
    safe = xp.maximum(ratio, 1e-16)
    return xp.clip(0.9 * safe ** (-0.2), 0.2, 5.0)


def integrate_dense(
    rhs: Rhs,
    state0: Array,
    lambda_total: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    initial_step: float = 0.05,
    max_step: float = 2.0,
) -> list[Array]:
    """Adaptively integrate a batch over an affine-parameter interval.

    Intended for analysis and tests where the whole trajectory is wanted;
    the ray tracer uses its own loop with termination conditions instead.

    Args:
        rhs: Right-hand side function, (n, d) to (n, d).
        state0: Initial states, shape (n, d).
        lambda_total: Total affine parameter to integrate.
        rtol: Relative tolerance for step acceptance.
        atol: Absolute tolerance for step acceptance.
        initial_step: Starting step size.
        max_step: Upper bound on the step size.

    Returns:
        List of accepted states, starting with a copy of state0. All
        samples in the batch share the step sequence (the most demanding
        sample controls the step), which is exact enough for test batches.
    """
    xp = xp_of(state0)
    n = state0.shape[0]
    state = state0.copy()
    states = [state.copy()]
    elapsed = 0.0
    h_scalar = float(initial_step)
    while elapsed < lambda_total:
        h_scalar = min(h_scalar, max_step, lambda_total - elapsed)
        h = xp.full((n,), h_scalar, dtype=state.dtype)
        new_state, error = dormand_prince_step(rhs, state, h)
        ratio = float(xp.max(error_ratio(state, new_state, error, rtol, atol)))
        if ratio <= 1.0:
            state = new_state
            states.append(state.copy())
            elapsed += h_scalar
        factor = float(min(max(0.9 * max(ratio, 1e-16) ** (-0.2), 0.2), 5.0))
        h_scalar *= factor
        if h_scalar < 1e-14:
            raise RuntimeError("integrate_dense: step size underflow")
    return states


def implicit_midpoint_step(
    rhs: Rhs,
    state: Array,
    h: Array,
    iterations: int = 4,
) -> Array:
    """One implicit midpoint step, symplectic for Hamiltonian flows.

    Solves y1 = y0 + h f((y0 + y1) / 2) by fixed-point iteration seeded
    with an explicit Euler half step. The midpoint rule is second order
    and symplectic, so the geodesic Hamiltonian shows bounded
    oscillation instead of the secular drift of explicit Runge-Kutta
    methods; use it for long orbital evolutions (debris streams,
    many-orbit trajectories) where energy fidelity over tens of
    thousands of steps matters more than order.

    Args:
        rhs: Derivative function mapping states (n, 8) to derivatives.
        state: Current states, shape (n, 8).
        h: Per-ray step sizes, shape (n,).
        iterations: Fixed-point iterations; four suffice for the step
            sizes used in practice (h well below the orbital timescale).

    Returns:
        Advanced states, shape (n, 8).
    """
    hh = h[:, None]
    midpoint = state + 0.5 * hh * rhs(state)
    for _ in range(iterations):
        midpoint = state + 0.5 * hh * rhs(midpoint)
    return state + hh * rhs(midpoint)
