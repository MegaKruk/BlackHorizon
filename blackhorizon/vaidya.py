"""Charged-Vaidya/Ori surrogate for the mass-inflation layer.

The realistic interior of a rotating hole develops mass inflation at
the Cauchy horizon: the crossflow of the infalling Price tail and an
outgoing flux drives the local Misner-Sharp mass to exponential
divergence while leaving only a weak null singularity, which an
observer crosses with finite integrated tidal distortion (Poisson and
Israel, Phys. Rev. D 41, 1796 (1990); Ori, Phys. Rev. Lett. 67, 789
(1991); Marolf and Ori, arXiv:1109.5139). The standard tractable
model is spherical: an ingoing charged Vaidya spacetime, whose charge
q stands in for Kerr's angular momentum by matching the horizon
structure r_pm = M pm sqrt(M^2 - q^2), crossed by a thin outgoing
null shell (the Ori model). This module implements that model
exactly, in Cartesian Kerr-Schild form so every tetrad, frame, and
imaging tool in this package applies unchanged:

    g = eta + 2 H l l,  l = (1, x_i / r),  v = t + r,
    H(v, r) = m(v, r) / r - q^2 / (2 r^2).

Ingredients:

- Price tail influx: m1(v) = M - dm_tail (v_tail / v)^(p-1) for
  v >= v_tail (held constant before), giving the luminosity
  L(v) = dm1/dv proportional to v^(-p) with p = 12 for the dominant
  quadrupole (Price 1972).
- Outgoing null shell: integrated exactly from dR/dv = f1(v, R)/2
  with f1 = 1 - 2 m1(v)/R + q^2/R^2; launched between the horizons it
  asymptotes the Cauchy horizon from above.
- Mass inflation: the null shell carries different advanced-time
  parametrizations on its two faces, dv2/dv1 = f1/f2, so the influx
  crossing it is received in region II with luminosity amplified by
  (f2/f1)^2, giving the matching ODE integrated exactly along the
  shell:

      dm2/dv = L(v) f2 / f1,   f_i = 1 - 2 m_i / R + q^2 / R^2,

  with m2(v0) = m1(v0) + E_shell. Because the shell rides the moving
  inner horizon of the evolving m1(v), |f1| along it tracks the tail
  luminosity, which cancels in the ODE and yields the classic
  Poisson-Israel-Ori result: m2 grows as exp(kappa_minus v) with a
  v^(-p) prefactor, kappa_minus = (r_plus - r_minus) / (2 r_minus^2)
  the inner-horizon surface gravity; the test suite verifies the
  measured e-folding rate against kappa_minus. Region II is labeled
  by the external advanced time in the single Kerr-Schild chart (the
  standard surrogate convention). The thin shell is smoothed over a
  small width for differentiable geometry, and the inflated mass is
  capped for numerics (the physics diverges; the cap is the render
  horizon).

The spacetime is genuinely time dependent, so p_t is no longer
conserved; vaidya_geodesic_rhs supplies the extended Hamiltonian flow
with dp_t/dlambda = (dH/dt) (l . p)^2. With the flux and shell off it
reduces to Reissner-Nordstrom and matches the stationary tracer
exactly, which the test suite verifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy

from .backend import Array, xp_of


@dataclass(frozen=True)
class VaidyaGeometry:
    """Pointwise geometry sample of the surrogate spacetime.

    Attributes:
        h: Kerr-Schild potential H, shape (n,).
        l: Spatial part of the ingoing null vector, shape (n, 3).
        h_spatial_scale: dH/dx_i = h_spatial_scale * l_i, shape (n,).
        h_t: dH/dt, shape (n,).
        radius: Areal radius, shape (n,).
    """

    h: Array
    l: Array
    h_spatial_scale: Array
    h_t: Array
    radius: Array


@dataclass
class ChargedVaidyaSpacetime:
    """The Ori mass-inflation surrogate in Cartesian Kerr-Schild form.

    Args:
        mass: Asymptotic mass M (geometric units).
        charge: Charge q in (0, M); choose q equal to the Kerr spin a
            being modeled to match the horizon structure.
        tail_mass: Total mass dm_tail carried by the Price tail.
        tail_start: Advanced time v_tail at which the tail turns on.
        tail_power: Price exponent p (luminosity v^-p), 12 for the
            dominant quadrupole.
        shell_energy: Asymptotic energy of the outgoing Ori shell.
        shell_start_v: Advanced time at which the shell is launched.
        shell_start_radius: Launch radius, strictly between the
            horizons.
        shell_width: Smoothing width of the thin shell.
        mass_cap: Numerical cap on the inflated mass function.
    """

    mass: float = 1.0
    charge: float = 0.9
    tail_mass: float = 0.01
    tail_start: float = 12.0
    tail_power: float = 12.0
    shell_energy: float = 1e-3
    shell_start_v: float = 14.0
    shell_start_radius: float = 1.0
    shell_width: float = 0.02
    mass_cap: float = 1e7
    time_dependent: bool = True
    _shell_v: numpy.ndarray = field(init=False, repr=False)
    _shell_r: numpy.ndarray = field(init=False, repr=False)
    _shell_m2: numpy.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.charge < self.mass:
            raise ValueError("charge must lie in (0, mass)")
        if not (
            self.inner_horizon_radius
            < self.shell_start_radius
            < self.outer_horizon_radius
        ):
            raise ValueError(
                "shell must launch strictly between the horizons"
            )
        self._integrate_shell()

    # Horizon structure of the asymptotic solution.

    @property
    def outer_horizon_radius(self) -> float:
        return self.mass + math.sqrt(self.mass**2 - self.charge**2)

    @property
    def inner_horizon_radius(self) -> float:
        return self.mass - math.sqrt(self.mass**2 - self.charge**2)

    @property
    def inner_surface_gravity(self) -> float:
        """kappa_minus = (r_plus - r_minus) / (2 r_minus^2)."""
        r_plus = self.outer_horizon_radius
        r_minus = self.inner_horizon_radius
        return (r_plus - r_minus) / (2.0 * r_minus**2)

    # Mass functions of the Ori construction.

    def ingoing_mass(self, v: Array) -> Array:
        """Region I mass m1(v): the Price-tail-fed exterior history."""
        xp = xp_of(v)
        v_safe = xp.maximum(v, self.tail_start)
        return self.mass - self.tail_mass * (
            self.tail_start / v_safe
        ) ** (self.tail_power - 1.0)

    def ingoing_mass_rate(self, v: Array) -> Array:
        """Price-tail luminosity L(v) = dm1/dv, zero before onset."""
        xp = xp_of(v)
        active = v >= self.tail_start
        rate = (
            self.tail_mass
            * (self.tail_power - 1.0)
            * self.tail_start ** (self.tail_power - 1.0)
            / xp.maximum(v, self.tail_start) ** self.tail_power
        )
        return xp.where(active, rate, 0.0)

    def metric_function(self, v: Array, radius: Array) -> Array:
        """f1 = 1 - 2 m1(v)/r + q^2/r^2 of region I."""
        return (
            1.0
            - 2.0 * self.ingoing_mass(v) / radius
            + self.charge**2 / radius**2
        )

    def _integrate_shell(self) -> None:
        """Co-integrate the shell trajectory and the inflated mass.

        dR/dv = f1/2 (outgoing null in region I; launched between the
        horizons where f1 < 0, the shell falls and rides the moving
        inner horizon of the evolving m1) together with the matching
        ODE dm2/dv = L(v) f2/f1 for the region II mass, m2(v0) =
        m1(v0) + E_shell. Midpoint steps; the table stops once m2 hits
        the numerical cap.
        """

        def f_of(mass, radius):
            return (
                1.0
                - 2.0 * mass / radius
                + self.charge**2 / radius**2
            )

        v = float(self.shell_start_v)
        radius = float(self.shell_start_radius)
        m2 = (
            float(self.ingoing_mass(numpy.array([v]))[0])
            + self.shell_energy
        )
        vs, rs, m2s = [v], [radius], [m2]
        step = 0.01
        for _ in range(4000000):
            m1 = float(self.ingoing_mass(numpy.array([v]))[0])
            lum = float(
                self.ingoing_mass_rate(numpy.array([v]))[0]
            )
            f1 = f_of(m1, radius)
            f2 = f_of(m2, radius)
            if m2 >= self.mass_cap or abs(f1) < 1e-300:
                break
            # Midpoint step of the coupled system.
            r_half = radius + 0.25 * step * f1
            m2_half = m2 + 0.5 * step * lum * f2 / f1
            v_half = v + 0.5 * step
            m1_half = float(
                self.ingoing_mass(numpy.array([v_half]))[0]
            )
            lum_half = float(
                self.ingoing_mass_rate(numpy.array([v_half]))[0]
            )
            f1_half = f_of(m1_half, r_half)
            f2_half = f_of(m2_half, r_half)
            radius += 0.5 * step * f1_half
            m2 += step * lum_half * f2_half / f1_half
            m2 = min(m2, self.mass_cap)
            v += step
            vs.append(v)
            rs.append(radius)
            m2s.append(m2)
        self._shell_v = numpy.asarray(vs)
        self._shell_r = numpy.asarray(rs)
        self._shell_m2 = numpy.asarray(m2s)

    def shell_radius(self, v: Array) -> Array:
        """Shell radius R_s(v); frozen at its endpoints outside the
        integrated table (before launch and after the CH asymptote)."""
        return numpy.interp(v, self._shell_v, self._shell_r)

    def inflated_mass_increment(self, v: Array) -> Array:
        """m2(v) - m1(v): the mass-inflation excess behind the shell.

        Interpolated from the co-integrated matching ODE; zero before
        the shell launch, frozen at its capped end value beyond the
        table (the render horizon of the divergence).
        """
        xp = xp_of(v)
        m2 = numpy.interp(v, self._shell_v, self._shell_m2)
        increment = m2 - self.ingoing_mass(v)
        return xp.where(
            v < self.shell_start_v,
            0.0,
            xp.clip(increment, 0.0, self.mass_cap),
        )

    def mass_function(self, v: Array, radius: Array) -> Array:
        """m(v, r): region I outside the shell, inflated region II
        inside, blended over the smoothing width."""
        xp = xp_of(v)
        shell_r = self.shell_radius(v)
        t = xp.clip(
            (shell_r - radius) / self.shell_width + 0.5, 0.0, 1.0
        )
        blend = t * t * (3.0 - 2.0 * t)
        return self.ingoing_mass(v) + blend * self.inflated_mass_increment(v)

    def misner_sharp_mass(self, v: Array, radius: Array) -> Array:
        """Quasilocal mass m(v, r) - q^2 / (2 r): the inflating
        diagnostic reported along observer worldlines."""
        return self.mass_function(v, radius) - self.charge**2 / (
            2.0 * radius
        )

    # Kerr-Schild geometry interface.

    def kerr_schild_radius(self, x: Array, y: Array, z: Array) -> Array:
        """Areal radius; the surrogate is spherical."""
        xp = xp_of(x)
        return xp.sqrt(x * x + y * y + z * z)

    def geometry_t(
        self, t: Array, x: Array, y: Array, z: Array
    ) -> VaidyaGeometry:
        """Geometry sample with time derivative, arrays of shape (n,).

        H(v, r) = m(v, r)/r - q^2/(2 r^2) with v = t + r; the spatial
        gradient is (H_v + H_r) l_i and the time derivative H_v, with
        the mass-function derivatives evaluated by small central
        differences (the shell table makes closed forms unwieldy; the
        stencil is far below the smoothing width).
        """
        xp = xp_of(x)
        radius = xp.maximum(self.kerr_schild_radius(x, y, z), 1e-9)
        l_vec = xp.stack(
            [x / radius, y / radius, z / radius], axis=-1
        )
        v = t + radius
        eps = 1e-4
        m_0 = self.mass_function(v, radius)
        m_v = (
            self.mass_function(v + eps, radius)
            - self.mass_function(v - eps, radius)
        ) / (2.0 * eps)
        m_r = (
            self.mass_function(v, radius + eps)
            - self.mass_function(v, radius - eps)
        ) / (2.0 * eps)
        h = m_0 / radius - self.charge**2 / (2.0 * radius**2)
        h_v = m_v / radius
        h_r = (
            m_r / radius
            - m_0 / radius**2
            + self.charge**2 / radius**3
        )
        return VaidyaGeometry(
            h=h,
            l=l_vec,
            h_spatial_scale=h_v + h_r,
            h_t=h_v,
            radius=radius,
        )

    def frozen(self, time: float) -> "FrozenVaidyaSlice":
        """Constant-time adapter exposing the stationary-style
        geometry(x, y, z) interface for the tetrad machinery, which is
        pointwise and needs no time derivative."""
        return FrozenVaidyaSlice(self, float(time))


class FrozenVaidyaSlice:
    """Duck-typed stationary view of the surrogate at fixed time."""

    def __init__(
        self, spacetime: ChargedVaidyaSpacetime, time: float
    ) -> None:
        self.spacetime = spacetime
        self.time = time
        self.inner_horizon_radius = spacetime.inner_horizon_radius
        self.outer_horizon_radius = spacetime.outer_horizon_radius

    def kerr_schild_radius(self, x, y, z):
        return self.spacetime.kerr_schild_radius(x, y, z)

    def geometry(self, x, y, z):
        xp = xp_of(x)
        sample = self.spacetime.geometry_t(
            xp.full_like(x, self.time), x, y, z
        )
        return sample


def vaidya_geodesic_rhs(
    spacetime: ChargedVaidyaSpacetime, states: Array
) -> Array:
    """Hamiltonian geodesic flow in the time-dependent surrogate.

    State rows are (t, x, y, z, p_t, p_x, p_y, p_z) with covariant
    momenta. With H_ham = (1/2)(-p_t^2 + p . p) - H (l . p_4)^2 and
    l . p_4 = -p_t + l . p:

        dt/dlambda   = -p_t + 2 H lp
        dx_i/dlambda = p_i - 2 H lp l_i
        dp_t/dlambda = (dH/dt) lp^2
        dp_i/dlambda = (dH/dx_i) lp^2
                       + 2 H lp (p_i - l_i (l . p)) / r,

    where the last term is the gradient of l for the radial null
    vector, dl_j/dx_i = (delta_ij - l_i l_j) / r. Reduces exactly to
    the stationary flow when the mass function is constant.
    """
    xp = xp_of(states)
    geo = spacetime.geometry_t(
        states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    )
    p_t = states[:, 4]
    p_s = states[:, 5:8]
    l_dot_p = xp.sum(geo.l * p_s, axis=-1)
    lp = -p_t + l_dot_p

    out = xp.empty_like(states)
    out[:, 0] = -p_t + 2.0 * geo.h * lp
    out[:, 1:4] = p_s - (2.0 * geo.h * lp)[:, None] * geo.l
    out[:, 4] = geo.h_t * lp**2
    grad_h = geo.h_spatial_scale[:, None] * geo.l
    l_gradient_term = (
        (2.0 * geo.h * lp / geo.radius)[:, None]
        * (p_s - l_dot_p[:, None] * geo.l)
    )
    out[:, 5:8] = grad_h * (lp**2)[:, None] + l_gradient_term
    return out
