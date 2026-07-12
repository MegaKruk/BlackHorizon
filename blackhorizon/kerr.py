"""Kerr spacetime in Cartesian Kerr-Schild coordinates.

The metric is written in the Kerr-Schild form

    g_mu_nu = eta_mu_nu + 2 H l_mu l_nu

with eta = diag(-1, 1, 1, 1), the scalar H = M r^3 / (r^4 + a^2 z^2) and the
null covector l_mu = (1, (r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2),
z / r). The Kerr-Schild radius r is the positive root of

    r^4 - (x^2 + y^2 + z^2 - a^2) r^2 - a^2 z^2 = 0.

These coordinates are horizon-penetrating: the metric is regular across the
event horizon, which is the property that makes stable ray tracing possible
(see docs/DESIGN.md section 2.2). Geometric units G = c = 1 are used, and by
default the mass sets the length unit (mass = 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .backend import Array, xp_of

_TINY = 1e-30
"""Guard against division by zero at the ring singularity. Large enough
that squaring a guarded quantity cannot underflow to zero in float64."""


@dataclass(frozen=True)
class KerrSchildGeometry:
    """Pointwise Kerr-Schild geometry evaluated at spatial positions.

    Attributes:
        radius: Kerr-Schild radial coordinate r, shape (...).
        h: Kerr-Schild scalar H, shape (...).
        l: Spatial part of the null covector l_i, shape (..., 3). The time
            component is identically 1 and is not stored.
        grad_h: Gradient dH/dx_i, shape (..., 3), or None if not requested.
        grad_l: Gradient of the spatial covector, shape (..., 3, 3) with
            grad_l[..., i, j] = d l_j / d x_i, or None if not requested.
    """

    radius: Array
    h: Array
    l: Array
    grad_h: Array | None = None
    grad_l: Array | None = None


class KerrSpacetime:
    """A rotating (Kerr) black hole described in Kerr-Schild coordinates.

    Responsible only for geometry: metric components, the Kerr-Schild
    scalar and null vector with their analytic gradients, and standard
    analytic radii (horizons, ISCO, photon orbits). Dynamics live in
    the geodesics module.
    """

    def __init__(self, mass: float = 1.0, spin: float = 0.0) -> None:
        """Create a Kerr spacetime.

        Args:
            mass: Black hole mass M in geometric units. Must be positive.
            spin: Angular momentum parameter a = J / M. Requires |a| <= M.
        """
        if mass <= 0.0:
            raise ValueError("mass must be positive")
        if abs(spin) > mass:
            raise ValueError("Kerr bound violated: |spin| must be <= mass")
        self.mass = float(mass)
        self.spin = float(spin)

    @property
    def outer_horizon_radius(self) -> float:
        """Radius of the outer event horizon r_plus."""
        m, a = self.mass, self.spin
        return m + math.sqrt(m * m - a * a)

    @property
    def inner_horizon_radius(self) -> float:
        """Radius of the inner (Cauchy) horizon r_minus."""
        m, a = self.mass, self.spin
        return m - math.sqrt(m * m - a * a)

    def ergosphere_radius(self, theta: float) -> float:
        """Outer ergosphere boundary at polar angle theta (radians)."""
        m, a = self.mass, self.spin
        return m + math.sqrt(m * m - a * a * math.cos(theta) ** 2)

    def isco_radius(self, prograde: bool = True) -> float:
        """Innermost stable circular orbit radius (Bardeen et al. 1972).

        Args:
            prograde: True for orbits corotating with the hole's spin.
        """
        chi = abs(self.spin) / self.mass
        z1 = 1.0 + (1.0 - chi * chi) ** (1.0 / 3.0) * (
            (1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0)
        )
        z2 = math.sqrt(3.0 * chi * chi + z1 * z1)
        root = math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
        sign = -1.0 if prograde else 1.0
        return self.mass * (3.0 + z2 + sign * root)

    def photon_orbit_radius(self, prograde: bool = True) -> float:
        """Equatorial circular photon orbit radius (Bardeen 1973)."""
        chi = abs(self.spin) / self.mass
        sign = -1.0 if prograde else 1.0
        angle = (2.0 / 3.0) * math.acos(sign * chi)
        return 2.0 * self.mass * (1.0 + math.cos(angle))

    def kerr_schild_radius(self, x: Array, y: Array, z: Array) -> Array:
        """Kerr-Schild radial coordinate r at Cartesian positions."""
        return self.geometry(x, y, z).radius

    def geometry(
        self, x: Array, y: Array, z: Array, gradients: bool = False
    ) -> KerrSchildGeometry:
        """Evaluate the Kerr-Schild scalar, null covector and gradients.

        Args:
            x, y, z: Cartesian coordinates, broadcastable arrays.
            gradients: If True, also compute the analytic gradients of H
                and of the spatial null covector needed by the geodesic
                equations of motion.

        Returns:
            A KerrSchildGeometry with fields on the broadcast shape.
        """
        xp = xp_of(x)
        m, a = self.mass, self.spin
        a2 = a * a

        rho2 = x * x + y * y + z * z
        q = rho2 - a2
        s = xp.sqrt(q * q + 4.0 * a2 * z * z)
        s = xp.maximum(s, _TINY)
        r2 = 0.5 * (q + s)
        r = xp.sqrt(xp.maximum(r2, _TINY))

        r2_plus_a2 = r2 + a2
        qd = xp.maximum(r2 * r2 + a2 * z * z, _TINY)
        h = m * r * r2 / qd

        inv_r2a2 = 1.0 / xp.maximum(r2_plus_a2, _TINY)
        inv_r = 1.0 / r
        lx = (r * x + a * y) * inv_r2a2
        ly = (r * y - a * x) * inv_r2a2
        lz = z * inv_r
        l = xp.stack([lx, ly, lz], axis=-1)

        if not gradients:
            return KerrSchildGeometry(radius=r, h=h, l=l)

        # Gradient of the Kerr-Schild radius from implicit differentiation.
        inv_s = 1.0 / s
        r_x = x * r * inv_s
        r_y = y * r * inv_s
        r_z = z * r2_plus_a2 * inv_r * inv_s

        # Gradient of H = M r^3 / (r^4 + a^2 z^2).
        inv_qd2 = 1.0 / (qd * qd)
        dh_dr = m * r2 * (3.0 * a2 * z * z - r2 * r2) * inv_qd2
        dh_dz_explicit = -2.0 * m * a2 * z * r * r2 * inv_qd2
        grad_h = xp.stack(
            [dh_dr * r_x, dh_dr * r_y, dh_dr * r_z + dh_dz_explicit],
            axis=-1,
        )

        # Gradients of the spatial null covector via the quotient rule.
        # For u = r x + a y and v = r y - a x over w = r^2 + a^2:
        # d(u/w)/dx_i = (du/dx_i) / w - (u / w) * (2 r dr/dx_i) / w.
        two_r = 2.0 * r
        dlx_dx = (r_x * x + r) * inv_r2a2 - lx * two_r * r_x * inv_r2a2
        dlx_dy = (r_y * x + a) * inv_r2a2 - lx * two_r * r_y * inv_r2a2
        dlx_dz = r_z * x * inv_r2a2 - lx * two_r * r_z * inv_r2a2
        dly_dx = (r_x * y - a) * inv_r2a2 - ly * two_r * r_x * inv_r2a2
        dly_dy = (r_y * y + r) * inv_r2a2 - ly * two_r * r_y * inv_r2a2
        dly_dz = r_z * y * inv_r2a2 - ly * two_r * r_z * inv_r2a2
        dlz_dx = -lz * r_x * inv_r
        dlz_dy = -lz * r_y * inv_r
        dlz_dz = (1.0 - lz * r_z) * inv_r

        row_x = xp.stack([dlx_dx, dly_dx, dlz_dx], axis=-1)
        row_y = xp.stack([dlx_dy, dly_dy, dlz_dy], axis=-1)
        row_z = xp.stack([dlx_dz, dly_dz, dlz_dz], axis=-1)
        grad_l = xp.stack([row_x, row_y, row_z], axis=-2)

        return KerrSchildGeometry(
            radius=r, h=h, l=l, grad_h=grad_h, grad_l=grad_l
        )

    def _null_covector_4(self, geo: KerrSchildGeometry) -> Array:
        """Assemble the full covector l_mu = (1, l_x, l_y, l_z)."""
        xp = xp_of(geo.l)
        ones = xp.ones_like(geo.h)
        return xp.concatenate([ones[..., None], geo.l], axis=-1)

    def metric(self, positions: Array) -> Array:
        """Covariant metric g_mu_nu at positions of shape (..., 4).

        Positions are spacetime points (t, x, y, z); the metric is
        stationary so t is ignored.
        """
        xp = xp_of(positions)
        geo = self.geometry(
            positions[..., 1], positions[..., 2], positions[..., 3]
        )
        l4 = self._null_covector_4(geo)
        eta = xp.zeros(positions.shape[:-1] + (4, 4), dtype=positions.dtype)
        eta[..., 0, 0] = -1.0
        for i in range(1, 4):
            eta[..., i, i] = 1.0
        outer = l4[..., :, None] * l4[..., None, :]
        return eta + 2.0 * geo.h[..., None, None] * outer

    def inverse_metric(self, positions: Array) -> Array:
        """Contravariant metric g^mu^nu at positions of shape (..., 4)."""
        xp = xp_of(positions)
        geo = self.geometry(
            positions[..., 1], positions[..., 2], positions[..., 3]
        )
        l4 = self._null_covector_4(geo)
        # Raise the index with eta: l^mu = (-1, l_x, l_y, l_z).
        l4_up = xp.concatenate([-l4[..., :1], l4[..., 1:]], axis=-1)
        eta_inv = xp.zeros(
            positions.shape[:-1] + (4, 4), dtype=positions.dtype
        )
        eta_inv[..., 0, 0] = -1.0
        for i in range(1, 4):
            eta_inv[..., i, i] = 1.0
        outer = l4_up[..., :, None] * l4_up[..., None, :]
        return eta_inv - 2.0 * geo.h[..., None, None] * outer
