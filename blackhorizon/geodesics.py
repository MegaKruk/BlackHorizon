"""Geodesic dynamics in Kerr-Schild coordinates.

State layout convention used throughout Black Horizon: a batch of geodesics
is an array of shape (n, 8) holding

    state[:, 0:4] = (t, x, y, z)          spacetime position
    state[:, 4:8] = (p_t, p_x, p_y, p_z)  covariant 4-momentum

The dynamics derive from the super-Hamiltonian

    H_ham = (1/2) g^mu^nu p_mu p_nu
          = (1/2) eta^mu^nu p_mu p_nu - H (l^mu p_mu)^2

which is conserved along geodesics (0 for photons, -1/2 for unit-mass
particles). See docs/DESIGN.md section 2.3 for the derivation of the
equations of motion.
"""

from __future__ import annotations

from .backend import Array, xp_of
from .kerr import KerrSpacetime


def build_state(positions: Array, momenta: Array, t0: float = 0.0) -> Array:
    """Assemble a (n, 8) state array from spatial positions and 4-momenta.

    Args:
        positions: Spatial positions (x, y, z), shape (n, 3).
        momenta: Covariant momenta (p_t, p_x, p_y, p_z), shape (n, 4).
        t0: Initial coordinate time assigned to every geodesic.
    """
    xp = xp_of(positions)
    n = positions.shape[0]
    state = xp.empty((n, 8), dtype=positions.dtype)
    state[:, 0] = t0
    state[:, 1:4] = positions
    state[:, 4:8] = momenta
    return state


def geodesic_rhs(spacetime: KerrSpacetime, state: Array) -> Array:
    """Right-hand side of Hamilton's equations for a batch of geodesics.

    Args:
        spacetime: The Kerr spacetime providing geometry and gradients.
        state: Geodesic states, shape (n, 8).

    Returns:
        d(state)/d(lambda), shape (n, 8). The p_t derivative is exactly
        zero because the spacetime is stationary.
    """
    xp = xp_of(state)
    geo = spacetime.geometry(
        state[:, 1], state[:, 2], state[:, 3], gradients=True
    )
    p_t = state[:, 4]
    p_s = state[:, 5:8]

    lp = -p_t + xp.sum(geo.l * p_s, axis=-1)
    two_h_lp = 2.0 * geo.h * lp

    deriv = xp.empty_like(state)
    deriv[:, 0] = -p_t + two_h_lp
    deriv[:, 1:4] = p_s - two_h_lp[:, None] * geo.l
    deriv[:, 4] = 0.0
    # dp_i/dlam = (dH/dx_i) lp^2 + 2 H lp sum_j p_j dl_j/dx_i
    # The contraction is written as an explicit broadcast-and-sum rather
    # than einsum: CuPy lowers this einsum to a cuBLAS batched matmul,
    # which the pip wheel does not bundle, and for a 3x3 contraction the
    # elementwise form is faster anyway.
    contraction = xp.sum(geo.grad_l * p_s[:, None, :], axis=-1)
    deriv[:, 5:8] = geo.grad_h * (lp * lp)[:, None] + two_h_lp[:, None] * contraction
    return deriv


def hamiltonian(spacetime: KerrSpacetime, state: Array) -> Array:
    """Super-Hamiltonian value per geodesic; conserved along the motion."""
    xp = xp_of(state)
    geo = spacetime.geometry(state[:, 1], state[:, 2], state[:, 3])
    p_t = state[:, 4]
    p_s = state[:, 5:8]
    lp = -p_t + xp.sum(geo.l * p_s, axis=-1)
    eta_term = 0.5 * (-p_t * p_t + xp.sum(p_s * p_s, axis=-1))
    return eta_term - geo.h * lp * lp


def conserved_quantities(state: Array) -> tuple[Array, Array]:
    """Energy E = -p_t and axial angular momentum L_z = x p_y - y p_x."""
    energy = -state[:, 4]
    l_z = state[:, 1] * state[:, 6] - state[:, 2] * state[:, 5]
    return energy, l_z


def coordinate_velocity(spacetime: KerrSpacetime, state: Array) -> Array:
    """Contravariant velocity dx^mu/dlambda = g^mu^nu p_nu, shape (n, 4)."""
    xp = xp_of(state)
    geo = spacetime.geometry(state[:, 1], state[:, 2], state[:, 3])
    p_t = state[:, 4]
    p_s = state[:, 5:8]
    lp = -p_t + xp.sum(geo.l * p_s, axis=-1)
    two_h_lp = 2.0 * geo.h * lp
    velocity = xp.empty(state.shape[:-1] + (4,), dtype=state.dtype)
    velocity[:, 0] = -p_t + two_h_lp
    velocity[:, 1:4] = p_s - two_h_lp[:, None] * geo.l
    return velocity


def null_momentum_from_velocity(
    spacetime: KerrSpacetime,
    positions: Array,
    directions: Array,
    time_orientation: str = "past",
    normalize: bool = True,
) -> Array:
    """Covariant photon momentum from a spatial propagation direction.

    Given a contravariant spatial direction s at each position, solve the
    null condition g_mu_nu k^mu k^nu = 0 for the time component k^t of the
    velocity k = (k^t, s), then lower the index: p_mu = g_mu_nu k^nu.

    Backward ray tracing from a camera uses the past-directed root
    (dt/dlambda < 0): integrating forward in the affine parameter then
    follows the received photon back in time, which orients Kerr
    frame-dragging correctly in rendered images.

    Args:
        spacetime: The Kerr spacetime.
        positions: Spatial positions, shape (n, 3). Must lie outside the
            ergosphere, where the quadratic has one root of each sign.
        directions: Contravariant spatial directions, shape (n, 3).
        time_orientation: 'past' or 'future'.
        normalize: If True, rescale so that |p_t| = 1 (affine
            reparametrization; sets the conserved energy magnitude to 1).

    Returns:
        Covariant momenta, shape (n, 4).
    """
    if time_orientation not in ("past", "future"):
        raise ValueError("time_orientation must be 'past' or 'future'")
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    h = geo.h
    l_dot_s = xp.sum(geo.l * directions, axis=-1)
    s_dot_s = xp.sum(directions * directions, axis=-1)

    # Quadratic in k^t: g_tt (k^t)^2 + 2 g_ti s_i k^t + g_ij s_i s_j = 0
    # with g_tt = 2H - 1, g_ti = 2H l_i, g_ij = delta_ij + 2H l_i l_j.
    g_tt = 2.0 * h - 1.0
    b_coef = 4.0 * h * l_dot_s
    c_coef = s_dot_s + 2.0 * h * l_dot_s * l_dot_s
    disc = b_coef * b_coef - 4.0 * g_tt * c_coef
    sqrt_disc = xp.sqrt(xp.maximum(disc, 0.0))
    root_a = (-b_coef + sqrt_disc) / (2.0 * g_tt)
    root_b = (-b_coef - sqrt_disc) / (2.0 * g_tt)
    if time_orientation == "future":
        k_t = xp.maximum(root_a, root_b)
    else:
        k_t = xp.minimum(root_a, root_b)

    # Lower the index: p_mu = g_mu_nu k^nu with l_mu k^mu = k^t + l . s.
    l_dot_k = k_t + l_dot_s
    momenta = xp.empty(positions.shape[:-1] + (4,), dtype=positions.dtype)
    momenta[:, 0] = -k_t + 2.0 * h * l_dot_k
    momenta[:, 1:4] = directions + (2.0 * h * l_dot_k)[:, None] * geo.l
    if normalize:
        momenta = momenta / xp.abs(momenta[:, 0])[:, None]
    return momenta


def timelike_momentum(
    spacetime: KerrSpacetime,
    positions: Array,
    coordinate_velocities: Array,
) -> Array:
    """Covariant momentum of a unit-mass particle from dx^i/dt.

    The 4-velocity is built as u^mu = u^t (1, v) with u^t fixed by the
    normalization g_mu_nu u^mu u^nu = -1, then lowered with the metric.

    Args:
        spacetime: The Kerr spacetime.
        positions: Spatial positions, shape (n, 3).
        coordinate_velocities: dx^i/dt, shape (n, 3). Must be subluminal
            at the given positions.

    Returns:
        Covariant momenta p_mu = g_mu_nu u^nu, shape (n, 4).
    """
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    h = geo.h
    v = coordinate_velocities
    l_dot_big_v = 1.0 + xp.sum(geo.l * v, axis=-1)
    norm2 = -1.0 + xp.sum(v * v, axis=-1) + 2.0 * h * l_dot_big_v * l_dot_big_v
    if bool(xp.any(norm2 >= 0.0)):
        raise ValueError(
            "coordinate velocity is not timelike at one or more positions"
        )
    u_t = 1.0 / xp.sqrt(-norm2)
    l_dot_u = u_t * l_dot_big_v
    momenta = xp.empty(positions.shape[:-1] + (4,), dtype=positions.dtype)
    momenta[:, 0] = -u_t + 2.0 * h * l_dot_u
    momenta[:, 1:4] = u_t[:, None] * v + (2.0 * h * l_dot_u)[:, None] * geo.l
    return momenta
