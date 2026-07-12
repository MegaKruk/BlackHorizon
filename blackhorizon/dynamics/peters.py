"""Gravitational-wave driven binary inspiral (Peters 1964).

Orbit-averaged evolution of the semi-major axis and eccentricity of a
two-body system radiating gravitational waves, and the circular-orbit
coalescence time, from Peters, P. C. 1964, Phys. Rev. 136, B1224.
Geometric units G = c = 1; masses and lengths share the same unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy


def coalescence_time_circular(m1: float, m2: float, separation: float) -> float:
    """Merger time of a circular binary: T = 5 a^4 / (256 m1 m2 M)."""
    total = m1 + m2
    return 5.0 * separation**4 / (256.0 * m1 * m2 * total)


def semi_major_axis_rate(m1: float, m2: float, a: float, e: float) -> float:
    """Orbit-averaged da/dt (Peters eq. 5.6)."""
    total = m1 + m2
    enhancement = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) / (
        1.0 - e**2
    ) ** 3.5
    return -(64.0 / 5.0) * m1 * m2 * total / a**3 * enhancement


def eccentricity_rate(m1: float, m2: float, a: float, e: float) -> float:
    """Orbit-averaged de/dt (Peters eq. 5.7)."""
    total = m1 + m2
    enhancement = (1.0 + (121.0 / 304.0) * e**2) / (1.0 - e**2) ** 2.5
    return -(304.0 / 15.0) * e * m1 * m2 * total / a**4 * enhancement


@dataclass(frozen=True)
class InspiralTrack:
    """Time series of an orbit-averaged inspiral.

    Attributes:
        times: Sample times, shape (n,).
        semi_major_axes: Semi-major axis at each time, shape (n,).
        eccentricities: Eccentricity at each time, shape (n,).
        merged: Whether the integration reached the merger cutoff.
    """

    times: numpy.ndarray
    semi_major_axes: numpy.ndarray
    eccentricities: numpy.ndarray
    merged: bool


def integrate_inspiral(
    m1: float,
    m2: float,
    a0: float,
    e0: float,
    a_merge: float | None = None,
    max_steps: int = 2000000,
) -> InspiralTrack:
    """Integrate the Peters equations until merger.

    Uses RK4 with an adaptive step tied to the instantaneous decay
    timescale, which shrinks smoothly through the final plunge.

    Args:
        m1: Primary mass.
        m2: Secondary mass.
        a0: Initial semi-major axis.
        e0: Initial eccentricity in [0, 1).
        a_merge: Separation treated as merged; defaults to six times the
            total mass (approximate ISCO scale).
        max_steps: Safety bound on integration steps.

    Returns:
        An InspiralTrack sampled at every accepted step.
    """
    if not 0.0 <= e0 < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    total = m1 + m2
    cutoff = 6.0 * total if a_merge is None else a_merge
    if a0 <= cutoff:
        raise ValueError("initial separation must exceed the merger cutoff")

    def rates(state):
        a, e = state
        e = max(e, 0.0)
        return numpy.array(
            [
                semi_major_axis_rate(m1, m2, a, e),
                eccentricity_rate(m1, m2, a, e),
            ]
        )

    state = numpy.array([a0, e0], dtype=float)
    t = 0.0
    times = [0.0]
    axes = [a0]
    eccs = [e0]
    merged = False
    for _ in range(max_steps):
        a, e = state
        if a <= cutoff:
            merged = True
            break
        timescale = a / abs(semi_major_axis_rate(m1, m2, a, max(e, 0.0)))
        h = 1e-3 * timescale
        k1 = rates(state)
        k2 = rates(state + 0.5 * h * k1)
        k3 = rates(state + 0.5 * h * k2)
        k4 = rates(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state[1] = min(max(state[1], 0.0), 0.999999)
        t += h
        times.append(t)
        axes.append(float(state[0]))
        eccs.append(float(state[1]))
    return InspiralTrack(
        times=numpy.asarray(times),
        semi_major_axes=numpy.asarray(axes),
        eccentricities=numpy.asarray(eccs),
        merged=merged,
    )
