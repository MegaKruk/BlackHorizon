"""Post-Newtonian N-body dynamics.

Accelerations for comparable-mass systems where test-particle geodesics
are inappropriate (see docs/DESIGN.md section 2.7):

- Newtonian gravity for any number of bodies.
- 1PN Einstein-Infeld-Hoffmann corrections (the standard n-body form of
  Newhall, Standish and Williams 1983, as used in ephemeris codes and
  REBOUNDx gr_full), giving relativistic periapsis precession.
- 2.5PN radiation reaction applied pairwise in the two-body form of
  Iyer and Will (1993), whose orbit-averaged energy and angular momentum
  losses reproduce the Peters (1964) equations.

Geometric units G = c = 1. State per body: position (3,), velocity (3,).
Deep strong-field regions are outside the domain of validity; use Stage 1
geodesics there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass(frozen=True)
class NBodySystem:
    """Masses and phase-space state of an N-body system.

    Attributes:
        masses: Body masses, shape (n,).
        positions: Cartesian positions, shape (n, 3).
        velocities: Cartesian velocities, shape (n, 3).
    """

    masses: numpy.ndarray
    positions: numpy.ndarray
    velocities: numpy.ndarray


def newtonian_acceleration(
    masses: numpy.ndarray, positions: numpy.ndarray
) -> numpy.ndarray:
    """Pairwise Newtonian accelerations, shape (n, 3)."""
    n = masses.shape[0]
    separation = positions[None, :, :] - positions[:, None, :]
    distance = numpy.linalg.norm(separation, axis=-1)
    numpy.fill_diagonal(distance, numpy.inf)
    inv_r3 = 1.0 / distance**3
    return numpy.einsum("j,ij,ijk->ik", masses, inv_r3, separation)


def eih_acceleration(
    masses: numpy.ndarray,
    positions: numpy.ndarray,
    velocities: numpy.ndarray,
) -> numpy.ndarray:
    """Newtonian plus 1PN Einstein-Infeld-Hoffmann accelerations.

    Implements the point-mass n-body equation of motion (Newhall,
    Standish and Williams 1983, eq. 1) with the accelerations appearing
    on the right-hand side evaluated at Newtonian order, the standard
    ephemeris practice.
    """
    n = masses.shape[0]
    newtonian = newtonian_acceleration(masses, positions)
    separation = positions[None, :, :] - positions[:, None, :]
    distance = numpy.linalg.norm(separation, axis=-1)
    numpy.fill_diagonal(distance, numpy.inf)
    inv_r = 1.0 / distance

    # Potentials at each body: sum_k m_k / r_ik.
    potential = numpy.sum(masses[None, :] * inv_r, axis=1)
    speed2 = numpy.sum(velocities * velocities, axis=-1)
    v_dot_v = velocities @ velocities.T

    acceleration = newtonian.copy()
    for i in range(n):
        correction = numpy.zeros(3)
        for j in range(n):
            if j == i:
                continue
            r_ij = distance[i, j]
            n_ij = separation[i, j] / r_ij
            radial_vj = float(n_ij @ velocities[j])
            bracket = (
                -4.0 * potential[i]
                - potential[j]
                + speed2[i]
                + 2.0 * speed2[j]
                - 4.0 * v_dot_v[i, j]
                - 1.5 * radial_vj**2
                + 0.5 * float(separation[i, j] @ newtonian[j])
            )
            correction += masses[j] / r_ij**2 * bracket * n_ij
            relative_velocity = velocities[i] - velocities[j]
            radial_mix = float(
                (-separation[i, j]) @ (4.0 * velocities[i] - 3.0 * velocities[j])
            )
            correction += (
                masses[j] / r_ij**3 * radial_mix * relative_velocity
            )
            correction += 3.5 * masses[j] / r_ij * newtonian[j]
        acceleration[i] += correction
    return acceleration


def radiation_reaction_acceleration(
    masses: numpy.ndarray,
    positions: numpy.ndarray,
    velocities: numpy.ndarray,
) -> numpy.ndarray:
    """Pairwise 2.5PN radiation-reaction accelerations (Iyer-Will 1993).

    For each pair, the relative acceleration is
    a = (8/5) (m1 m2 / r^3) [ (v . n) n (3 v^2 + 17 M / (3 r))
                              - v (v^2 + 3 M / r) ]
    with M the pair's total mass, distributed to the bodies with the mass
    ratio factors that leave the center of mass unaccelerated. Orbit
    averages reproduce the Peters (1964) da/dt and de/dt.
    """
    n = masses.shape[0]
    acceleration = numpy.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            total = masses[i] + masses[j]
            rel_position = positions[i] - positions[j]
            rel_velocity = velocities[i] - velocities[j]
            r = float(numpy.linalg.norm(rel_position))
            n_ij = rel_position / r
            v2 = float(rel_velocity @ rel_velocity)
            radial = float(rel_velocity @ n_ij)
            relative = (
                (8.0 / 5.0)
                * masses[i]
                * masses[j]
                / r**3
                * (
                    radial * (3.0 * v2 + 17.0 * total / (3.0 * r)) * n_ij
                    - (v2 + 3.0 * total / r) * rel_velocity
                )
            )
            acceleration[i] += masses[j] / total * relative
            acceleration[j] -= masses[i] / total * relative
    return acceleration


def total_acceleration(
    system: NBodySystem,
    include_1pn: bool = True,
    include_radiation: bool = True,
) -> numpy.ndarray:
    """Combined acceleration at the configured post-Newtonian order."""
    if include_1pn:
        acceleration = eih_acceleration(
            system.masses, system.positions, system.velocities
        )
    else:
        acceleration = newtonian_acceleration(
            system.masses, system.positions
        )
    if include_radiation:
        acceleration = acceleration + radiation_reaction_acceleration(
            system.masses, system.positions, system.velocities
        )
    return acceleration


def step_rk4(
    system: NBodySystem,
    dt: float,
    include_1pn: bool = True,
    include_radiation: bool = True,
) -> NBodySystem:
    """Advance the system one RK4 step of size dt."""

    def derivative(positions, velocities):
        state = NBodySystem(system.masses, positions, velocities)
        return velocities, total_acceleration(
            state, include_1pn, include_radiation
        )

    x0, v0 = system.positions, system.velocities
    k1x, k1v = derivative(x0, v0)
    k2x, k2v = derivative(x0 + 0.5 * dt * k1x, v0 + 0.5 * dt * k1v)
    k3x, k3v = derivative(x0 + 0.5 * dt * k2x, v0 + 0.5 * dt * k2v)
    k4x, k4v = derivative(x0 + dt * k3x, v0 + dt * k3v)
    return NBodySystem(
        masses=system.masses,
        positions=x0 + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
        velocities=v0 + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v),
    )


def newtonian_energy(system: NBodySystem) -> float:
    """Newtonian total energy, for conservation checks and diagnostics."""
    kinetic = 0.5 * float(
        numpy.sum(
            system.masses * numpy.sum(system.velocities**2, axis=-1)
        )
    )
    potential = 0.0
    n = system.masses.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r = float(
                numpy.linalg.norm(system.positions[i] - system.positions[j])
            )
            potential -= system.masses[i] * system.masses[j] / r
    return kinetic + potential
