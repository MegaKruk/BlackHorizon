"""Physics validation: conserved quantities, photon sphere, circular orbits."""

import numpy

from blackhorizon.geodesics import (
    build_state,
    conserved_quantities,
    geodesic_rhs,
    hamiltonian,
    null_momentum_from_velocity,
    timelike_momentum,
)
from blackhorizon.integrators import integrate_dense
from blackhorizon.kerr import KerrSpacetime


def kerr_schild_radii(st, states):
    return st.kerr_schild_radius(states[:, 1], states[:, 2], states[:, 3])


class TestConservation:
    def test_photon_strong_field_flyby(self):
        """E, L_z and the Hamiltonian stay constant through a close flyby."""
        st = KerrSpacetime(spin=0.9)
        positions = numpy.array([[30.0, 4.0, 1.5]])
        directions = numpy.array([[-1.0, 0.0, 0.0]])
        momenta = null_momentum_from_velocity(
            st, positions, directions, time_orientation="future"
        )
        state0 = build_state(positions, momenta)

        def rhs(batch):
            return geodesic_rhs(st, batch)

        states = integrate_dense(rhs, state0, lambda_total=70.0, rtol=1e-11)
        trajectory = numpy.concatenate(states, axis=0)
        energy, l_z = conserved_quantities(trajectory)
        ham = hamiltonian(st, trajectory)

        numpy.testing.assert_allclose(energy, energy[0], rtol=1e-12)
        numpy.testing.assert_allclose(l_z, l_z[0], rtol=1e-8)
        numpy.testing.assert_allclose(ham, 0.0, atol=1e-9)

        # The flyby must actually probe the strong field to be meaningful.
        radii = kerr_schild_radii(st, trajectory)
        assert float(radii.min()) < 6.0

    def test_timelike_normalization_conserved(self):
        """For a massive particle H_ham = -1/2 along the whole orbit."""
        st = KerrSpacetime(spin=0.9)
        r_orbit = 6.0
        omega = 1.0 / (r_orbit ** 1.5 + st.spin)
        positions = numpy.array([[r_orbit, st.spin, 0.0]])
        velocities = numpy.array([[-omega * st.spin, omega * r_orbit, 0.0]])
        momenta = timelike_momentum(st, positions, velocities)
        state0 = build_state(positions, momenta)

        def rhs(batch):
            return geodesic_rhs(st, batch)

        states = integrate_dense(rhs, state0, lambda_total=120.0, rtol=1e-11)
        trajectory = numpy.concatenate(states, axis=0)
        ham = hamiltonian(st, trajectory)
        numpy.testing.assert_allclose(ham, -0.5, rtol=1e-9)


class TestPhotonSphere:
    def test_tangential_photon_stays_on_sphere(self):
        """A tangential photon at r = 3M rides the unstable circular orbit."""
        st = KerrSpacetime(spin=0.0)
        positions = numpy.array([[3.0, 0.0, 0.0]])
        directions = numpy.array([[0.0, 1.0, 0.0]])
        momenta = null_momentum_from_velocity(
            st, positions, directions, time_orientation="future"
        )
        state0 = build_state(positions, momenta)

        def rhs(batch):
            return geodesic_rhs(st, batch)

        # The orbit is unstable: perturbations grow by roughly e^(2 pi)
        # per revolution, so integrate about 1.5 orbits. With E = 1 the
        # affine length of one orbit is 2 pi / (d phi / d lambda) with
        # d phi / d lambda = b / r^2 = sqrt(3) / 3, about 10.9 M.
        states = integrate_dense(rhs, state0, lambda_total=16.0, rtol=1e-12)
        trajectory = numpy.concatenate(states, axis=0)
        radii = kerr_schild_radii(st, trajectory)
        numpy.testing.assert_allclose(radii, 3.0, atol=1e-6)
        # Confirm it actually went around: the azimuth must wind.
        phi = numpy.unwrap(numpy.arctan2(trajectory[:, 2], trajectory[:, 1]))
        assert abs(phi[-1] - phi[0]) > 2.5 * numpy.pi

    def test_critical_impact_parameter_of_tangential_photon(self):
        """The tangential photon at 3M has b = L_z / E = 3 sqrt(3) M."""
        st = KerrSpacetime(spin=0.0)
        positions = numpy.array([[3.0, 0.0, 0.0]])
        directions = numpy.array([[0.0, 1.0, 0.0]])
        momenta = null_momentum_from_velocity(
            st, positions, directions, time_orientation="future"
        )
        state0 = build_state(positions, momenta)
        energy, l_z = conserved_quantities(state0)
        b = float(l_z[0] / energy[0])
        numpy.testing.assert_allclose(b, 3.0 * numpy.sqrt(3.0), rtol=1e-12)


class TestCircularOrbit:
    def test_massive_circular_orbit_stays_circular(self):
        """A prograde circular orbit at r = 6M around a = 0.9 stays put.

        The Boyer-Lindquist angular velocity Omega = 1 / (r^1.5 + a) carries
        over to Kerr-Schild azimuth for constant r. The orbit point (r, a, 0)
        follows from x + i y = (r + i a) exp(i phi) in the equatorial plane.
        """
        st = KerrSpacetime(spin=0.9)
        r_orbit = 6.0
        assert r_orbit > st.isco_radius(prograde=True)
        omega = 1.0 / (r_orbit ** 1.5 + st.spin)
        positions = numpy.array([[r_orbit, st.spin, 0.0]])
        velocities = numpy.array([[-omega * st.spin, omega * r_orbit, 0.0]])
        momenta = timelike_momentum(st, positions, velocities)
        state0 = build_state(positions, momenta)

        def rhs(batch):
            return geodesic_rhs(st, batch)

        # Roughly two orbital periods of proper time.
        states = integrate_dense(rhs, state0, lambda_total=170.0, rtol=1e-11)
        trajectory = numpy.concatenate(states, axis=0)
        radii = kerr_schild_radii(st, trajectory)
        numpy.testing.assert_allclose(radii, r_orbit, rtol=1e-6)
        assert numpy.allclose(trajectory[:, 3], 0.0, atol=1e-8)

        energy, l_z = conserved_quantities(trajectory)
        numpy.testing.assert_allclose(energy, energy[0], rtol=1e-12)
        numpy.testing.assert_allclose(l_z, l_z[0], rtol=1e-10)
