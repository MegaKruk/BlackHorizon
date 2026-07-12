"""Dynamics tests: Peters inspiral, PN N-body, tidal disruption."""

import numpy
import pytest

from blackhorizon.dynamics.peters import (
    coalescence_time_circular,
    eccentricity_rate,
    integrate_inspiral,
    semi_major_axis_rate,
)
from blackhorizon.dynamics.pn_nbody import (
    NBodySystem,
    newtonian_energy,
    step_rk4,
)
from blackhorizon.dynamics.tde import (
    energy_spread_estimate,
    fallback_rate,
    generate_debris_stream,
    hills_mass,
    tidal_radius,
)
from blackhorizon.geodesics import geodesic_rhs, hamiltonian
from blackhorizon.integrators import rk4_step
from blackhorizon.kerr import KerrSpacetime


class TestPeters:
    def test_circular_track_matches_closed_form(self):
        """a(t)^4 = a0^4 - (256/5) m1 m2 M t for circular inspirals."""
        m1, m2, a0 = 1.0, 1.0, 60.0
        track = integrate_inspiral(m1, m2, a0, 0.0, a_merge=20.0)
        assert track.merged
        beta = 256.0 / 5.0 * m1 * m2 * (m1 + m2)
        expected = (a0**4 - beta * track.times) ** 0.25
        numpy.testing.assert_allclose(
            track.semi_major_axes, expected, rtol=1e-3
        )

    def test_coalescence_time_consistency(self):
        """The integrated merger time matches the analytic T formula."""
        m1, m2, a0 = 1.0, 0.5, 50.0
        track = integrate_inspiral(m1, m2, a0, 0.0, a_merge=1.0)
        analytic = coalescence_time_circular(m1, m2, a0)
        assert abs(track.times[-1] - analytic) / analytic < 0.02

    def test_eccentric_binaries_merge_faster(self):
        circular = integrate_inspiral(1.0, 1.0, 60.0, 0.0, a_merge=15.0)
        eccentric = integrate_inspiral(1.0, 1.0, 60.0, 0.6, a_merge=15.0)
        assert eccentric.times[-1] < 0.25 * circular.times[-1]

    def test_eccentricity_decays(self):
        assert eccentricity_rate(1.0, 1.0, 50.0, 0.5) < 0.0
        assert eccentricity_rate(1.0, 1.0, 50.0, 0.0) == 0.0


class TestPnNbody:
    def test_newtonian_energy_conserved(self):
        omega = numpy.sqrt(2.0 / 40.0**3)
        system = NBodySystem(
            masses=numpy.array([1.0, 1.0]),
            positions=numpy.array([[-20.0, 0, 0], [20.0, 0, 0]], dtype=float),
            velocities=numpy.array(
                [[0, -20.0 * omega, 0], [0, 20.0 * omega, 0]], dtype=float
            ),
        )
        e0 = newtonian_energy(system)
        dt = 2 * numpy.pi / omega / 400
        for _ in range(800):
            system = step_rk4(
                system, dt, include_1pn=False, include_radiation=False
            )
        assert abs(newtonian_energy(system) - e0) / abs(e0) < 1e-8

    def test_1pn_periapsis_precession(self):
        """Test-mass precession matches 6 pi M / (a (1 - e^2)) per orbit."""
        m, a_orbit, e = 1.0, 200.0, 0.3
        r_peri = a_orbit * (1.0 - e)
        v_peri = numpy.sqrt(m * (1.0 + e) / (a_orbit * (1.0 - e)))
        system = NBodySystem(
            masses=numpy.array([m, 1e-9]),
            positions=numpy.array(
                [[0.0, 0.0, 0.0], [r_peri, 0.0, 0.0]]
            ),
            velocities=numpy.array(
                [[0.0, 0.0, 0.0], [0.0, v_peri, 0.0]]
            ),
        )
        period = 2.0 * numpy.pi * numpy.sqrt(a_orbit**3 / m)
        dt = period / 4000
        radii, angles = [], []
        for _ in range(3 * 4000 + 400):
            system = step_rk4(system, dt, include_radiation=False)
            rel = system.positions[1] - system.positions[0]
            radii.append(float(numpy.linalg.norm(rel)))
            angles.append(float(numpy.arctan2(rel[1], rel[0])))
        radii = numpy.asarray(radii)
        angles = numpy.unwrap(numpy.asarray(angles))
        minima = (
            numpy.where(
                (radii[1:-1] < radii[:-2]) & (radii[1:-1] < radii[2:])
            )[0]
            + 1
        )
        assert len(minima) >= 2
        precession = (angles[minima[1]] - angles[minima[0]]) - 2 * numpy.pi
        expected = 6.0 * numpy.pi * m / (a_orbit * (1.0 - e**2))
        assert abs(precession - expected) / expected < 0.02

    def test_radiation_reaction_matches_peters_circular(self):
        """Semi-major axis decay follows the closed-form Peters solution."""
        m1, m2, sep = 1.0, 1.0, 60.0
        total = m1 + m2
        omega = numpy.sqrt(total / sep**3)
        system = NBodySystem(
            masses=numpy.array([m1, m2]),
            positions=numpy.array(
                [[-sep / 2, 0, 0], [sep / 2, 0, 0]], dtype=float
            ),
            velocities=numpy.array(
                [[0, -sep / 2 * omega, 0], [0, sep / 2 * omega, 0]],
                dtype=float,
            ),
        )

        def orbital_a(state):
            mu = m1 * m2 / total
            rel_v = state.velocities[0] - state.velocities[1]
            rel_x = state.positions[0] - state.positions[1]
            energy = 0.5 * mu * float(rel_v @ rel_v) - m1 * m2 / float(
                numpy.linalg.norm(rel_x)
            )
            return -m1 * m2 / (2.0 * energy)

        period = 2 * numpy.pi / omega
        dt = period / 600
        steps = int(30 * period / dt)
        state = system
        for _ in range(steps):
            state = step_rk4(
                state, dt, include_1pn=False, include_radiation=True
            )
        elapsed = steps * dt
        beta = 256.0 / 5.0 * m1 * m2 * total
        expected = (sep**4 - beta * elapsed) ** 0.25
        assert abs(orbital_a(state) - expected) / expected < 0.01

    def test_center_of_mass_drifts_inertially(self):
        """Radiation reaction leaves the Newtonian center of mass
        inertial: the pairwise mass-ratio distribution must cancel
        exactly. The 1PN terms are excluded because EIH is Lorentz
        rather than Galilean invariant, so a boosted Newtonian center
        of mass is not its conserved quantity."""
        m1, m2, sep = 2.0, 1.0, 40.0
        total = m1 + m2
        omega = numpy.sqrt(total / sep**3)
        drift = numpy.array([0.01, 0.0, 0.0])
        system = NBodySystem(
            masses=numpy.array([m1, m2]),
            positions=numpy.array(
                [[-sep * m2 / total, 0, 0], [sep * m1 / total, 0, 0]]
            ),
            velocities=numpy.array(
                [
                    [0, -sep * m2 / total * omega, 0],
                    [0, sep * m1 / total * omega, 0],
                ]
            )
            + drift[None, :],
        )
        period = 2 * numpy.pi / omega
        dt = period / 400
        steps = 1000
        state = system
        for _ in range(steps):
            state = step_rk4(state, dt, include_1pn=False)
        com0 = (system.masses @ system.positions) / total
        expected = com0 + drift * steps * dt
        com1 = (state.masses @ state.positions) / total
        numpy.testing.assert_allclose(com1, expected, atol=1e-5)


class TestTde:
    def test_tidal_radius_scaling(self):
        base = tidal_radius(1.0, 1e-6, 0.01)
        assert tidal_radius(8.0, 1e-6, 0.01) == pytest.approx(2.0 * base)
        assert tidal_radius(1.0, 8e-6, 0.01) == pytest.approx(base / 2.0)

    def test_hills_mass_marginal_case(self):
        """At the Hills mass the tidal radius equals 2 M."""
        star_mass, star_radius = 1e-6, 0.02
        m_hills = hills_mass(star_mass, star_radius)
        scaled_radius = star_radius / m_hills
        scaled_mass = star_mass / m_hills
        r_t = tidal_radius(1.0, scaled_mass, scaled_radius)
        assert abs(r_t - 2.0) < 1e-9

    def test_fallback_slope_and_normalization(self):
        times = numpy.geomspace(1.0, 1000.0, 4000)
        for partial, slope in ((False, -5.0 / 3.0), (True, -9.0 / 4.0)):
            rate = fallback_rate(times, 1.0, 0.5, partial=partial)
            measured = numpy.log(rate[2000] / rate[1000]) / numpy.log(
                times[2000] / times[1000]
            )
            assert abs(measured - slope) < 1e-6
            total = numpy.trapezoid(rate, times)
            assert abs(total - 0.5) / 0.5 < 0.05
        assert fallback_rate(0.5, 1.0, 0.5) == 0.0

    def test_debris_stream_energy_spread(self):
        st = KerrSpacetime(spin=0.9)
        stream = generate_debris_stream(
            st, star_mass=1e-6, star_radius=0.5, n_particles=4000, seed=3
        )
        energies = stream.specific_energies
        # Roughly half the debris is bound for a parabolic encounter.
        assert 0.3 < stream.bound_fraction < 0.7
        spread = float(energies.std())
        analytic = energy_spread_estimate(1.0, 0.5, stream.tidal_radius)
        assert 0.1 * analytic < spread < 3.0 * analytic

    def test_debris_evolves_on_valid_geodesics(self):
        """Debris states integrate with conserved timelike Hamiltonian."""
        st = KerrSpacetime(spin=0.9)
        stream = generate_debris_stream(
            st, star_mass=1e-6, star_radius=0.5, n_particles=64, seed=5
        )
        state = stream.states
        h = numpy.full((state.shape[0],), 0.5)

        def rhs(batch):
            return geodesic_rhs(st, batch)

        for _ in range(400):
            state = rk4_step(rhs, state, h)
        values = hamiltonian(st, state)
        numpy.testing.assert_allclose(values, -0.5, atol=1e-5)
