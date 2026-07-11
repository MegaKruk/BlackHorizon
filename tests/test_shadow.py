"""End-to-end shadow validation via capture/escape boundaries.

These tests exercise the whole chain (initial conditions, geodesic RHS,
adaptive integration, termination) against exact general-relativistic
results: the Schwarzschild critical impact parameter b_c = 3 sqrt(3) M and
the frame-dragging asymmetry of the Kerr shadow.
"""

import numpy

from blackhorizon.geodesics import (
    build_state,
    conserved_quantities,
    null_momentum_from_velocity,
)
from blackhorizon.kerr import KerrSpacetime
from blackhorizon.tracer import RayStatus, trace_rays

DISTANCE = 1000.0


def trace_equatorial_fan(spacetime, aim_offsets):
    """Trace photons from a distant equatorial camera toward the hole.

    Rays start at (DISTANCE, 0, 0) with contravariant direction
    (-1, offset, 0) normalized: they travel inward, offset in the +y or -y
    direction, staying in the equatorial plane. Returns per-ray status and
    the exact impact parameter b = L_z / E from the conserved quantities.
    """
    n = aim_offsets.shape[0]
    positions = numpy.tile(numpy.array([[DISTANCE, 0.0, 0.0]]), (n, 1))
    directions = numpy.stack(
        [-numpy.ones(n), aim_offsets, numpy.zeros(n)], axis=-1
    )
    directions /= numpy.linalg.norm(directions, axis=-1, keepdims=True)
    momenta = null_momentum_from_velocity(
        spacetime, positions, directions, time_orientation="past"
    )
    state0 = build_state(positions, momenta)
    result = trace_rays(
        spacetime,
        state0,
        escape_radius=1.2 * DISTANCE,
        max_steps=60000,
        rtol=1e-10,
        atol=1e-13,
        max_step=5.0,
    )
    energy, l_z = conserved_quantities(state0)
    # b = L_z / E is invariant under the past-directed sign convention.
    impact = l_z / energy
    return result.status, impact


def capture_boundary(spacetime, offset_low, offset_high, rounds=3, fan=48):
    """Bisect the capture/escape boundary; returns |b| at the boundary.

    Scans a fan of aim offsets, finds the largest captured and smallest
    escaped offsets, and refines the bracket. Requires the boundary to be
    monotone in the scanned interval, which holds for equatorial fans on
    one side of the hole.
    """
    low, high = offset_low, offset_high
    boundary_b = None
    for _ in range(rounds):
        offsets = numpy.linspace(low, high, fan)
        status, impact = trace_equatorial_fan(spacetime, offsets)
        captured = status == int(RayStatus.CAPTURED)
        escaped = status == int(RayStatus.ESCAPED)
        assert bool(captured.any()), "bracket contains no captured rays"
        assert bool(escaped.any()), "bracket contains no escaped rays"
        assert bool((captured | escaped).all()), "unresolved rays in fan"
        magnitudes = numpy.abs(offsets)
        idx_cap = int(numpy.argmax(numpy.where(captured, magnitudes, -1.0)))
        low = float(offsets[idx_cap])
        esc_above = escaped & (magnitudes > magnitudes[idx_cap])
        idx_esc = int(
            numpy.argmin(numpy.where(esc_above, magnitudes, numpy.inf))
        )
        high = float(offsets[idx_esc])
        boundary_b = 0.5 * float(
            numpy.abs(impact[idx_cap]) + numpy.abs(impact[idx_esc])
        )
    return boundary_b


class TestSchwarzschildShadow:
    def test_critical_impact_parameter(self):
        """The capture boundary sits at b_c = 3 sqrt(3) M within 0.5 percent."""
        st = KerrSpacetime(spin=0.0)
        b_measured = capture_boundary(st, 4.5e-3, 6.5e-3)
        b_exact = 3.0 * numpy.sqrt(3.0)
        assert abs(b_measured - b_exact) / b_exact < 5e-3


def analytic_critical_impact_parameter(st: KerrSpacetime, prograde: bool):
    """|b| of the equatorial circular photon orbit (Bardeen 1973).

    xi = (r^2 (r - 3M) + a^2 (r + M)) / (a (M - r)) evaluated at the
    photon orbit radius gives the critical L_z / E; the capture boundary
    for equatorial photons sits at |xi|.
    """
    m, a = st.mass, st.spin
    r = st.photon_orbit_radius(prograde=prograde)
    xi = (r * r * (r - 3.0 * m) + a * a * (r + m)) / (a * (m - r))
    return abs(xi)


class TestKerrShadowAsymmetry:
    def test_prograde_retrograde_asymmetry(self):
        """Frame dragging shrinks the prograde side of the shadow.

        For spin a = 0.9 the measured capture boundaries must match the
        analytic critical impact parameters (about 2.84 M prograde and
        6.83 M retrograde) within 0.5 percent.
        """
        st = KerrSpacetime(spin=0.9)
        b_pro_exact = analytic_critical_impact_parameter(st, prograde=True)
        b_ret_exact = analytic_critical_impact_parameter(st, prograde=False)
        # With b = L_z / E = -D * offset for this camera, negative aim
        # offsets give prograde photons (L_z > 0 with the +z spin).
        b_prograde = capture_boundary(st, -4.0e-3, -2.0e-3)
        b_retrograde = capture_boundary(st, 5.5e-3, 8.0e-3)
        assert abs(b_prograde - b_pro_exact) / b_pro_exact < 5e-3
        assert abs(b_retrograde - b_ret_exact) / b_ret_exact < 5e-3
        assert b_prograde < b_retrograde

    def test_zero_spin_is_symmetric(self):
        st = KerrSpacetime(spin=0.0)
        b_plus = capture_boundary(st, 4.5e-3, 6.5e-3, rounds=2)
        b_minus = capture_boundary(st, -6.5e-3, -4.5e-3, rounds=2)
        assert abs(b_plus - b_minus) / b_plus < 1e-3


class TestTracerBehavior:
    def test_radial_infall_is_captured_and_outward_escapes(self):
        st = KerrSpacetime(spin=0.9)
        positions = numpy.array([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        directions = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        momenta = null_momentum_from_velocity(
            st, positions, directions, time_orientation="future"
        )
        state0 = build_state(positions, momenta)
        result = trace_rays(st, state0, escape_radius=30.0)
        assert int(result.status[0]) == int(RayStatus.CAPTURED)
        assert int(result.status[1]) == int(RayStatus.ESCAPED)

    def test_escape_radius_validation(self):
        st = KerrSpacetime()
        positions = numpy.array([[50.0, 0.0, 0.0]])
        directions = numpy.array([[-1.0, 0.0, 0.0]])
        momenta = null_momentum_from_velocity(st, positions, directions)
        state0 = build_state(positions, momenta)
        try:
            trace_rays(st, state0, escape_radius=40.0)
        except ValueError:
            return
        raise AssertionError("expected ValueError for bad escape radius")
