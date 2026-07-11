"""Integrator tests on a system with a known exact solution."""

import numpy

from blackhorizon.integrators import (
    dormand_prince_step,
    error_ratio,
    rk4_step,
)


def harmonic_rhs(state):
    """Simple harmonic oscillator: d(q, p)/dt = (p, -q)."""
    deriv = numpy.empty_like(state)
    deriv[:, 0] = state[:, 1]
    deriv[:, 1] = -state[:, 0]
    return deriv


def integrate_fixed(step_fn, state, h_scalar, n_steps):
    h = numpy.full((state.shape[0],), h_scalar)
    for _ in range(n_steps):
        out = step_fn(harmonic_rhs, state, h)
        state = out[0] if isinstance(out, tuple) else out
    return state


def exact_solution(t):
    """Solution for initial condition (q, p) = (1, 0)."""
    return numpy.array([[numpy.cos(t), -numpy.sin(t)]])


def test_rk4_fourth_order_convergence():
    t_final = 2.0 * numpy.pi
    errors = []
    for n_steps in (100, 200, 400):
        state = numpy.array([[1.0, 0.0]])
        state = integrate_fixed(rk4_step, state, t_final / n_steps, n_steps)
        errors.append(numpy.max(numpy.abs(state - exact_solution(t_final))))
    order1 = numpy.log2(errors[0] / errors[1])
    order2 = numpy.log2(errors[1] / errors[2])
    assert 3.7 < order1 < 4.3
    assert 3.7 < order2 < 4.3


def test_dormand_prince_fifth_order_convergence():
    t_final = 2.0 * numpy.pi
    errors = []
    for n_steps in (50, 100, 200):
        state = numpy.array([[1.0, 0.0]])
        state = integrate_fixed(
            dormand_prince_step, state, t_final / n_steps, n_steps
        )
        errors.append(numpy.max(numpy.abs(state - exact_solution(t_final))))
    order1 = numpy.log2(errors[0] / errors[1])
    order2 = numpy.log2(errors[1] / errors[2])
    assert 4.6 < order1 < 5.4
    assert 4.6 < order2 < 5.4


def test_error_estimate_tracks_true_error():
    state = numpy.array([[1.0, 0.0]])
    h = numpy.array([0.3])
    new_state, err = dormand_prince_step(harmonic_rhs, state, h)
    true_err = numpy.max(numpy.abs(new_state - exact_solution(0.3)))
    est_err = numpy.max(numpy.abs(err))
    # The embedded estimate must be within two orders of magnitude of truth.
    assert est_err > 0.0
    assert 1e-2 < est_err / max(true_err, 1e-300) < 1e2


def test_error_ratio_flags_nan_as_infinite():
    state = numpy.array([[1.0, 0.0]])
    bad = numpy.array([[numpy.nan, 0.0]])
    ratio = error_ratio(state, state, bad, rtol=1e-9, atol=1e-12)
    assert numpy.isinf(ratio[0])
