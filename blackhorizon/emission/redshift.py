"""Relativistic redshift of disk emission.

Computes the total redshift factor g = nu_obs / nu_em for photons
received from matter on prograde circular equatorial orbits, combining
gravitational redshift, Doppler shift, and relativistic aberration in
one covariant expression g = (p . u_obs) / (p . u_em). This module is
the float64 mirror of the corresponding shader code and is what the
tests validate against analytic results.
"""

from __future__ import annotations

import numpy

from ..backend import Array, xp_of
from ..kerr import KerrSpacetime


def circular_orbit_velocity(
    spacetime: KerrSpacetime, positions: Array
) -> Array:
    """Coordinate velocity dx^i/dt of prograde circular equatorial orbits.

    In Kerr-Schild Cartesian coordinates a circular orbit rotates the
    position rigidly about the spin axis: v = Omega (-y, x, 0) with the
    Boyer-Lindquist angular velocity Omega = 1 / (r^(3/2) + a).

    Args:
        spacetime: The Kerr spacetime.
        positions: Equatorial positions (x, y, z ~ 0), shape (n, 3).
    """
    xp = xp_of(positions)
    r = spacetime.kerr_schild_radius(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    omega = 1.0 / (r**1.5 + spacetime.spin)
    velocity = xp.empty_like(positions)
    velocity[:, 0] = -omega * positions[:, 1]
    velocity[:, 1] = omega * positions[:, 0]
    velocity[:, 2] = 0.0
    return velocity


def emitter_four_velocity(
    spacetime: KerrSpacetime, positions: Array
) -> tuple[Array, Array]:
    """Normalized 4-velocity (u^t, u^i) of circular-orbit disk matter."""
    xp = xp_of(positions)
    velocity = circular_orbit_velocity(spacetime, positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    l_dot_v = 1.0 + xp.sum(geo.l * velocity, axis=-1)
    norm2 = (
        -1.0
        + xp.sum(velocity * velocity, axis=-1)
        + 2.0 * geo.h * l_dot_v * l_dot_v
    )
    u_t = 1.0 / xp.sqrt(-norm2)
    return u_t, u_t[:, None] * velocity


def static_observer_lapse(
    spacetime: KerrSpacetime, position: numpy.ndarray
) -> float:
    """u^t of a static observer at a position (the camera).

    Static observers exist outside the ergosphere where g_tt < 0; the
    factor converts photon energies to frequencies the camera measures.
    """
    geo = spacetime.geometry(
        numpy.asarray([position[0]]),
        numpy.asarray([position[1]]),
        numpy.asarray([position[2]]),
    )
    g_tt = 2.0 * float(geo.h[0]) - 1.0
    if g_tt >= 0.0:
        raise ValueError("camera inside the ergosphere has no static frame")
    return 1.0 / numpy.sqrt(-g_tt)


def redshift_factor(
    spacetime: KerrSpacetime,
    hit_positions: Array,
    traced_momenta: Array,
    observer_lapse: float = 1.0,
) -> Array:
    """Redshift g = nu_obs / nu_em for traced camera rays hitting the disk.

    The tracer follows past-directed rays; the physical photon
    momentum is their negative, so with the emitter 4-velocity u the
    emitted frequency is nu_em = -(p_phys . u) = p_t u^t +
    p_spatial . u_spatial (traced components). For exterior static
    cameras rays carry p_t = +1 and nu_obs = observer_lapse; for
    tetrad-built infalling cameras every ray has unit camera frequency
    by construction, so observer_lapse is exactly one and p_t varies
    per ray.

    Args:
        spacetime: The Kerr spacetime.
        hit_positions: Disk crossing points, shape (n, 3).
        traced_momenta: Covariant traced momenta (p_t, p_i) at the
            crossing, shape (n, 4), with p_t = +1.
        observer_lapse: u^t of the static camera; 1 for a camera at
            infinity.

    Returns:
        Redshift factors, shape (n,). Values above one are blueshifted
        (Doppler-boosted approaching side), below one redshifted.
    """
    xp = xp_of(hit_positions)
    u_t, u_spatial = emitter_four_velocity(spacetime, hit_positions)
    frequency_emitted = traced_momenta[:, 0] * u_t + xp.sum(
        traced_momenta[:, 1:4] * u_spatial, axis=-1
    )
    return observer_lapse / frequency_emitted
