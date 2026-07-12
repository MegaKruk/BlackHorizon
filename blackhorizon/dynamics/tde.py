"""Tidal disruption events.

Analytic prescriptions (see docs/DESIGN.md section 2.6) plus a
test-particle debris stream generator whose particles evolve on the
validated Stage 1 Kerr geodesics:

- Tidal radius r_t = R_star (M_bh / M_star)^(1/3) (Hills 1975; Rees 1988).
- Hills mass: the largest hole that disrupts a star outside the horizon.
- Fallback rate: t^(-5/3) for full disruptions (Rees 1988), steepening to
  t^(-9/4) for partial disruptions (Coughlin and Nixon 2019).
- Frozen-in debris energy spread Delta epsilon ~ M_bh R_star / r_t^2
  across the stellar diameter at disruption.

Geometric units G = c = 1; the black hole mass of the supplied spacetime
sets the length unit as elsewhere in Black Horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from ..geodesics import build_state, timelike_momentum
from ..kerr import KerrSpacetime


def _marginally_bound_speed(
    spacetime: KerrSpacetime,
    position: numpy.ndarray,
    direction: numpy.ndarray,
) -> float:
    """Coordinate speed making the conserved energy E = -p_t exactly 1.

    This is the general-relativistic analogue of the Newtonian parabolic
    speed sqrt(2 M / r): a particle launched with it is marginally bound.
    Solved by bisection since E grows monotonically with speed.
    """

    def energy(speed: float) -> float:
        """Conserved energy at a trial speed; infinite outside the
        local light cone so bisection stays within it."""
        velocity = (speed * direction)[None, :]
        try:
            momentum = timelike_momentum(
                spacetime, position[None, :], velocity
            )
        except ValueError:
            return numpy.inf
        return float(-momentum[0, 0])

    low, high = 1e-6, 0.999999
    if energy(low) >= 1.0:
        raise ValueError("even a static particle is unbound here")
    for _ in range(80):
        mid = 0.5 * (low + high)
        if energy(mid) < 1.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def tidal_radius(bh_mass: float, star_mass: float, star_radius: float) -> float:
    """Tidal disruption radius r_t = R (M / m)^(1/3)."""
    return star_radius * (bh_mass / star_mass) ** (1.0 / 3.0)


def hills_mass(star_mass: float, star_radius: float) -> float:
    """Largest black hole mass that disrupts the star outside r = 2 M.

    Solves r_t(M) = 2 M for M. In geometric units
    M = (star_radius / 2)^(3/2) / star_mass^(1/2); for a Sun-like star
    this evaluates to roughly 1e8 solar masses.
    """
    return (star_radius / 2.0) ** 1.5 / star_mass**0.5


def fallback_rate(
    times, peak_time: float, disrupted_mass: float, partial: bool = False
):
    """Mass fallback rate after a disruption.

    Args:
        times: Times since disruption, array or scalar.
        peak_time: Return time of the most bound debris; the rate is zero
            before it.
        disrupted_mass: Total mass in bound debris.
        partial: If True use the partial-disruption decay t^(-9/4)
            (Coughlin and Nixon 2019) instead of the classic t^(-5/3).

    Returns:
        dM/dt with the normalization such that the integral from
        peak_time to infinity equals disrupted_mass.
    """
    t = numpy.asarray(times, dtype=float)
    exponent = -9.0 / 4.0 if partial else -5.0 / 3.0
    # Integral of (t / t0)^p from t0 to infinity is t0 / (-p - 1).
    normalization = disrupted_mass * (-exponent - 1.0) / peak_time
    rate = normalization * (t / peak_time) ** exponent
    return numpy.where(t >= peak_time, rate, 0.0)


@dataclass(frozen=True)
class DebrisStream:
    """Initial conditions of a tidally disrupted star.

    Attributes:
        states: Geodesic states (n, 8) ready for Stage 1 integration.
        specific_energies: Conserved E = -p_t per particle, shape (n,).
        bound_fraction: Fraction of particles with E < 1 (bound debris).
        tidal_radius: The disruption radius used.
    """

    states: numpy.ndarray
    specific_energies: numpy.ndarray
    bound_fraction: float
    tidal_radius: float


def generate_debris_stream(
    spacetime: KerrSpacetime,
    star_mass: float,
    star_radius: float,
    n_particles: int = 2000,
    penetration: float = 1.0,
    prograde: bool = True,
    seed: int = 0,
) -> DebrisStream:
    """Disrupt a star at pericenter into geodesic test particles.

    The star arrives on a parabolic orbit with pericenter
    r_p = r_t / penetration. At pericenter each fluid element's orbit is
    frozen in: particles sample the stellar sphere, share the center of
    mass velocity, and their conserved energies inherit the tidal spread
    from their radial offset. Self-gravity of the debris is neglected,
    a documented approximation.

    Args:
        spacetime: The Kerr spacetime (black hole mass 1).
        star_mass: Stellar mass in units of the hole mass.
        star_radius: Stellar radius in units of M.
        n_particles: Number of test particles.
        penetration: beta = r_t / r_p; 1 grazes the tidal radius.
        prograde: Orbit sense relative to the hole spin.
        seed: Random seed for the particle sampling.

    Returns:
        A DebrisStream with ready-to-integrate geodesic states.
    """
    r_t = tidal_radius(1.0, star_mass, star_radius)
    r_p = r_t / penetration
    if r_p < 4.0 * spacetime.outer_horizon_radius:
        raise ValueError(
            "pericenter too close to the horizon for the test-particle "
            "prescription; reduce penetration or star compactness"
        )

    rng = numpy.random.default_rng(seed)
    offsets = rng.normal(scale=star_radius / 2.0, size=(n_particles, 3))
    radii = numpy.linalg.norm(offsets, axis=-1)
    keep = radii <= star_radius
    offsets[~keep] *= (
        star_radius / numpy.maximum(radii[~keep], 1e-12) * 0.99
    )[:, None]

    # Center of mass at pericenter on the x axis, moving tangentially,
    # prograde or retrograde in the equator, at the relativistically
    # marginally bound (parabolic) speed so E_com = 1 exactly.
    center = numpy.array([r_p, 0.0, 0.0])
    direction = numpy.array([0.0, 1.0 if prograde else -1.0, 0.0])
    speed = _marginally_bound_speed(spacetime, center, direction)
    positions = center[None, :] + offsets
    velocities = numpy.tile(speed * direction, (n_particles, 1))

    momenta = timelike_momentum(spacetime, positions, velocities)
    states = build_state(positions, momenta)
    energies = -states[:, 4]
    bound_fraction = float(numpy.mean(energies < 1.0))
    return DebrisStream(
        states=states,
        specific_energies=energies,
        bound_fraction=bound_fraction,
        tidal_radius=r_t,
    )


def energy_spread_estimate(
    bh_mass: float, star_radius: float, r_t: float
) -> float:
    """Frozen-in specific energy spread Delta epsilon = M R / r_t^2."""
    return bh_mass * star_radius / r_t**2
