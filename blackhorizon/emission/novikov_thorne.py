"""Novikov-Thorne relativistic thin accretion disk.

Implements the closed-form Page and Thorne (1974) flux profile for a
geometrically thin, optically thick disk around a Kerr black hole, with
the zero-torque inner boundary at the ISCO. The effective temperature
follows from Stefan-Boltzmann, T proportional to F^(1/4); absolute
normalization is a free scale in this simulator (the peak temperature is
a user-facing parameter), so the profile functions return shapes, not
Kelvin. Geometric units G = c = M = 1; radii in units of M.

Known model limitation (documented in docs/DESIGN.md): the profile is
truncated at the ISCO; emission from the plunging region is neglected.
"""

from __future__ import annotations

import math

import numpy

from ..kerr import KerrSpacetime


def _radicand_roots(spin: float) -> tuple[float, float, float]:
    """Roots x1, x2, x3 of x^3 - 3 x + 2 a = 0 (x = sqrt(r/M))."""
    angle = math.acos(max(-1.0, min(1.0, spin))) / 3.0
    x1 = 2.0 * math.cos(angle - math.pi / 3.0)
    x2 = 2.0 * math.cos(angle + math.pi / 3.0)
    x3 = -2.0 * math.cos(angle)
    return x1, x2, x3


def page_thorne_flux(radii, spin: float):
    """Dimensionless Page-Thorne flux profile F(r).

    Args:
        radii: Boyer-Lindquist radii in units of M, array or scalar.
        spin: Black hole spin a/M in [-1, 1].

    Returns:
        Flux in arbitrary units, zero at and inside the ISCO. The shape
        matches the Page and Thorne (1974) closed form; overall scale is
        arbitrary.
    """
    r = numpy.asarray(radii, dtype=float)
    spacetime = KerrSpacetime(mass=1.0, spin=spin)
    r_isco = spacetime.isco_radius(prograde=True)
    x = numpy.sqrt(numpy.maximum(r, 1e-12))
    x0 = math.sqrt(r_isco)
    x1, x2, x3 = _radicand_roots(spin)

    bracket = x - x0 - 1.5 * spin * numpy.log(
        numpy.maximum(x / x0, 1e-30)
    )
    for xi, xj, xk in ((x1, x2, x3), (x2, x3, x1), (x3, x1, x2)):
        if abs(xi) < 1e-12:
            # The coefficient carries (xi - a)^2 / xi which vanishes in
            # the a -> 0 limit where a root sits at zero.
            continue
        coefficient = (
            3.0 * (xi - spin) ** 2 / (xi * (xi - xj) * (xi - xk))
        )
        ratio = (x - xi) / (x0 - xi)
        bracket = bracket - coefficient * numpy.log(
            numpy.maximum(ratio, 1e-30)
        )

    denominator = x**4 * (x**3 - 3.0 * x + 2.0 * spin)
    flux = numpy.where(
        r > r_isco,
        1.5 * bracket / numpy.maximum(denominator, 1e-30),
        0.0,
    )
    return numpy.maximum(flux, 0.0)


def temperature_profile(radii, spin: float):
    """Dimensionless effective temperature T(r) = F(r)^(1/4)."""
    return page_thorne_flux(radii, spin) ** 0.25


def temperature_lut(
    spin: float, outer_radius: float, size: int = 512
) -> tuple[numpy.ndarray, float, float]:
    """Normalized temperature lookup table between the ISCO and r_out.

    Args:
        spin: Black hole spin a/M.
        outer_radius: Outer disk radius in units of M; must exceed the
            ISCO radius.
        size: Number of table entries.

    Returns:
        Tuple (table, r_inner, r_outer) where table has shape (size,)
        with values in [0, 1] (1 at the temperature peak) sampled
        linearly in radius between r_inner (the ISCO) and r_outer.
    """
    spacetime = KerrSpacetime(mass=1.0, spin=spin)
    r_inner = spacetime.isco_radius(prograde=True)
    if outer_radius <= r_inner:
        raise ValueError("outer_radius must exceed the ISCO radius")
    radii = numpy.linspace(r_inner, outer_radius, size)
    profile = temperature_profile(radii, spin)
    peak = profile.max()
    if peak <= 0.0:
        raise RuntimeError("temperature profile unexpectedly vanished")
    return (profile / peak).astype(numpy.float32), float(r_inner), float(
        outer_radius
    )


def peak_temperature_radius(spin: float) -> float:
    """Radius (units of M) where the disk temperature peaks."""
    spacetime = KerrSpacetime(mass=1.0, spin=spin)
    r_inner = spacetime.isco_radius(prograde=True)
    radii = numpy.linspace(r_inner * 1.0001, r_inner * 20.0, 20000)
    profile = temperature_profile(radii, spin)
    return float(radii[int(numpy.argmax(profile))])
