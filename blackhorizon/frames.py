"""Observer frames: infalling 4-velocities, orthonormal tetrads, rays.

Inside the horizon no static observers exist (the time Killing vector
is spacelike there), so the camera must ride a timelike worldline and
generate rays by local aberration from its orthonormal tetrad. This
module provides:

- The rain (Doran) 4-velocity: free fall from rest at infinity with
  E = 1 and l-aligned momentum, the Kerr generalization of the
  Painleve-Gullstrand observer. In ingoing Kerr-Schild coordinates its
  covariant momentum is p = (-1, w l) with w = -sqrt(2H)/(1+sqrt(2H)),
  regular through both horizons (Doran, arXiv:gr-qc/9910099; Hamilton
  and Lisle, Am. J. Phys. 76:519, 2008).
- Metric-orthonormal tetrads built by Gram-Schmidt from any timelike
  4-velocity and a desired viewing orientation.
- Past-directed traced camera-ray momenta from a tetrad, normalized so
  the camera-frame frequency of every ray is exactly one; the
  covariant redshift then needs no observer lapse.

All routines are array-module generic (NumPy or CuPy).
"""

from __future__ import annotations

import numpy

from .backend import Array, xp_of
from .kerr import KerrSpacetime

_ETA_DIAG = (-1.0, 1.0, 1.0, 1.0)


def metric_dot(
    spacetime: KerrSpacetime, positions: Array, a_vec: Array, b_vec: Array
) -> Array:
    """Inner product g(A, B) of contravariant 4-vectors at positions.

    Uses g = eta + 2 H l l with covariant l = (1, l_i), so
    g(A, B) = eta(A, B) + 2 H (A^t + l . A_s)(B^t + l . B_s).

    Args:
        spacetime: The Kerr spacetime.
        positions: Points, shape (n, 3).
        a_vec: Contravariant 4-vectors, shape (n, 4).
        b_vec: Contravariant 4-vectors, shape (n, 4).

    Returns:
        g(A, B), shape (n,).
    """
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    eta = (
        -a_vec[:, 0] * b_vec[:, 0]
        + xp.sum(a_vec[:, 1:4] * b_vec[:, 1:4], axis=-1)
    )
    l_a = a_vec[:, 0] + xp.sum(geo.l * a_vec[:, 1:4], axis=-1)
    l_b = b_vec[:, 0] + xp.sum(geo.l * b_vec[:, 1:4], axis=-1)
    return eta + 2.0 * geo.h * l_a * l_b


def lower_index(
    spacetime: KerrSpacetime, positions: Array, vectors: Array
) -> Array:
    """Lower a contravariant 4-vector: v_mu = g_mu_nu v^nu."""
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    l_v = vectors[:, 0] + xp.sum(geo.l * vectors[:, 1:4], axis=-1)
    lowered = xp.empty_like(vectors)
    lowered[:, 0] = -vectors[:, 0] + 2.0 * geo.h * l_v
    lowered[:, 1:4] = vectors[:, 1:4] + (2.0 * geo.h * l_v)[
        :, None
    ] * geo.l
    return lowered


def raise_index(
    spacetime: KerrSpacetime, positions: Array, covectors: Array
) -> Array:
    """Raise a covariant 4-vector: v^mu = g^mu_nu v_nu.

    Uses g^-1 = eta - 2 H l^ l^ with contravariant l^ = (-1, l_i).
    """
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    l_up_v = -covectors[:, 0] + xp.sum(
        geo.l * covectors[:, 1:4], axis=-1
    )
    raised = xp.empty_like(covectors)
    raised[:, 0] = -covectors[:, 0] + 2.0 * geo.h * l_up_v
    raised[:, 1:4] = covectors[:, 1:4] - (2.0 * geo.h * l_up_v)[
        :, None
    ] * geo.l
    return raised


def rain_four_velocity(
    spacetime: KerrSpacetime, positions: Array
) -> Array:
    """Contravariant 4-velocity of the rain (Doran) observer.

    Free fall from rest at infinity, E = 1, with spatial momentum along
    the Kerr-Schild null vector l. Regular through both horizons; at
    large radius it reduces to an observer at rest.

    Args:
        spacetime: The Kerr spacetime.
        positions: Points, shape (n, 3).

    Returns:
        u^mu, shape (n, 4), future-directed and normalized g(u,u) = -1.
    """
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    root = xp.sqrt(xp.maximum(2.0 * geo.h, 0.0))
    w = -root / (1.0 + root)
    covariant = xp.empty(positions.shape[:-1] + (4,), dtype=positions.dtype)
    covariant[:, 0] = -1.0
    covariant[:, 1:4] = w[:, None] * geo.l
    return raise_index(spacetime, positions, covariant)


def build_tetrad(
    spacetime: KerrSpacetime,
    position: numpy.ndarray,
    four_velocity: numpy.ndarray,
    forward: numpy.ndarray,
    up: numpy.ndarray,
) -> numpy.ndarray:
    """Orthonormal tetrad for one observer by metric Gram-Schmidt.

    Args:
        position: Observer position, shape (3,).
        four_velocity: Contravariant timelike 4-velocity, shape (4,).
        forward: Desired viewing direction seed (spatial), shape (3,).
        up: Desired up seed (spatial), shape (3,).

    Returns:
        Tetrad e, shape (4, 4), rows (e0, e1, e2, e3) contravariant
        with e0 the 4-velocity, e1 the view direction, e2 right, e3 up,
        satisfying g(e_a, e_b) = eta_ab.
    """
    pos = numpy.asarray(position, dtype=float)[None, :]

    def dot(a, b):
        return float(
            metric_dot(spacetime, pos, a[None, :], b[None, :])[0]
        )

    e0 = numpy.asarray(four_velocity, dtype=float)
    norm0 = dot(e0, e0)
    if norm0 >= 0.0:
        raise ValueError("observer 4-velocity must be timelike")
    e0 = e0 / numpy.sqrt(-norm0)

    seeds = [
        numpy.concatenate([[0.0], numpy.asarray(forward, dtype=float)]),
        numpy.concatenate(
            [[0.0], numpy.cross(forward, up).astype(float)]
        ),
        numpy.concatenate([[0.0], numpy.asarray(up, dtype=float)]),
    ]
    basis = [e0]
    for seed in seeds:
        vector = seed + dot(seed, e0) * e0
        for spatial in basis[1:]:
            vector = vector - dot(seed, spatial) * spatial
        norm = dot(vector, vector)
        if norm <= 1e-14:
            raise ValueError("degenerate tetrad seed directions")
        basis.append(vector / numpy.sqrt(norm))
    return numpy.stack(basis)


def tetrad_ray_momenta(
    spacetime: KerrSpacetime,
    position: numpy.ndarray,
    tetrad: numpy.ndarray,
    local_directions: numpy.ndarray,
) -> numpy.ndarray:
    """Past-directed traced covariant momenta for tetrad camera rays.

    A photon arriving from local sky direction n has physical momentum
    p_phys = e0 - (n1 e1 + n2 e2 + n3 e3), null with camera-frame
    frequency -g(p_phys, e0) = 1 by construction. The traced momentum
    is its negative lowered, so the backward ray leaves the camera
    along the look direction and every ray carries unit camera
    frequency; the covariant redshift then reads
    g = 1 / (p_traced . u_emitter).

    Args:
        spacetime: The Kerr spacetime.
        position: Camera position, shape (3,).
        tetrad: Orthonormal tetrad from build_tetrad, shape (4, 4).
        local_directions: Unit look directions in the local frame
            (components along e1, e2, e3), shape (n, 3).

    Returns:
        Traced covariant momenta (p_t, p_x, p_y, p_z), shape (n, 4).
    """
    n = local_directions.shape[0]
    spatial = (
        local_directions[:, 0:1] * tetrad[1][None, :]
        + local_directions[:, 1:2] * tetrad[2][None, :]
        + local_directions[:, 2:3] * tetrad[3][None, :]
    )
    physical = tetrad[0][None, :] - spatial
    positions = numpy.tile(
        numpy.asarray(position, dtype=float)[None, :], (n, 1)
    )
    return lower_index(spacetime, positions, -physical)
