"""Charged-Vaidya/Ori mass-inflation surrogate (Stage 7).

The surrogate is a spherical charged-Vaidya spacetime crossed by an
outgoing Ori shell, in Cartesian Kerr-Schild form. Tests anchor it to
analytic results: Reissner-Nordstrom horizon structure matching Kerr,
the exact Hamiltonian flow of the time-dependent metric, reduction to
the stationary case when the fluxes vanish, and the Poisson-Israel-Ori
mass-inflation e-folding rate equal to the inner-horizon surface
gravity.
"""

import numpy

from blackhorizon.integrators import rk4_step
from blackhorizon.vaidya import (
    ChargedVaidyaSpacetime,
    vaidya_geodesic_rhs,
)


def _hamiltonian(spacetime, states):
    geo = spacetime.geometry_t(
        states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    )
    p_t = states[:, 4]
    p_s = states[:, 5:8]
    lp = -p_t + numpy.sum(geo.l * p_s, axis=-1)
    return 0.5 * (-p_t**2 + numpy.sum(p_s * p_s, axis=-1)) - geo.h * lp**2


class TestSurrogateStructure:
    def test_horizons_match_kerr(self):
        """Charge q matches Kerr spin a: r_pm = M pm sqrt(M^2 - q^2)."""
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        assert abs(spacetime.outer_horizon_radius - 1.435889) < 1e-5
        assert abs(spacetime.inner_horizon_radius - 0.564110) < 1e-5

    def test_inner_surface_gravity(self):
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        r_plus = spacetime.outer_horizon_radius
        r_minus = spacetime.inner_horizon_radius
        expected = (r_plus - r_minus) / (2.0 * r_minus**2)
        assert abs(spacetime.inner_surface_gravity - expected) < 1e-12

    def test_shell_reaches_cauchy_horizon(self):
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        assert (
            abs(spacetime._shell_r[-1] - spacetime.inner_horizon_radius)
            < 1e-3
        )
        assert spacetime._shell_r[0] > spacetime.inner_horizon_radius


class TestHamiltonianFlow:
    def test_rhs_matches_numerical_flow(self):
        """The RHS equals Hamilton's equations of the extended
        Hamiltonian to central-difference accuracy."""
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        state = numpy.array(
            [[20.0, 0.9, 0.3, 0.2, -1.1, 0.4, -0.7, 0.5]]
        )
        rhs = vaidya_geodesic_rhs(spacetime, state)
        eps = 1e-6
        for k in range(4):
            plus = state.copy()
            plus[0, k] += eps
            minus = state.copy()
            minus[0, k] -= eps
            dp = -(
                _hamiltonian(spacetime, plus)
                - _hamiltonian(spacetime, minus)
            )[0] / (2.0 * eps)
            assert abs(dp - rhs[0, 4 + k]) < 1e-6
        for k in range(4):
            plus = state.copy()
            plus[0, 4 + k] += eps
            minus = state.copy()
            minus[0, 4 + k] -= eps
            dx = (
                _hamiltonian(spacetime, plus)
                - _hamiltonian(spacetime, minus)
            )[0] / (2.0 * eps)
            assert abs(dx - rhs[0, k]) < 1e-6

    def test_static_limit_conserves_energy(self):
        """With fluxes off the metric is stationary: p_t and the
        Hamiltonian are conserved along a ray."""
        spacetime = ChargedVaidyaSpacetime(
            charge=0.9, tail_mass=0.0, shell_energy=0.0
        )
        state = numpy.array(
            [[0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.55, 0.0]]
        )
        p_t0 = float(state[0, 4])
        h0 = float(_hamiltonian(spacetime, state)[0])

        def rhs(batch):
            return vaidya_geodesic_rhs(spacetime, batch)

        for _ in range(20000):
            state = rk4_step(rhs, state, numpy.array([0.002]))
        assert abs(float(state[0, 4]) - p_t0) < 1e-9
        assert abs(float(_hamiltonian(spacetime, state)[0]) - h0) < 1e-9

    def test_time_dependence_evolves_energy(self):
        """With the Price tail on, photon energy is not conserved."""
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        state = numpy.array(
            [[13.0, 3.0, 0.0, 0.5, 1.0, -0.9, 0.2, 0.1]]
        )
        p_t0 = float(state[0, 4])

        def rhs(batch):
            return vaidya_geodesic_rhs(spacetime, batch)

        for _ in range(3000):
            state = rk4_step(rhs, state, numpy.array([0.002]))
        assert abs(float(state[0, 4]) - p_t0) > 1e-4


class TestMassInflation:
    def test_efolding_rate_matches_surface_gravity(self):
        """The inflated mass grows as v^(-p) exp(kappa_minus v): the
        classic Poisson-Israel-Ori result. Fitting ln(m2) + p ln(v)
        against v recovers kappa_minus."""
        spacetime = ChargedVaidyaSpacetime(charge=0.9, mass_cap=1e12)
        kappa = spacetime.inner_surface_gravity
        power = spacetime.tail_power
        v = spacetime._shell_v
        increment = spacetime._shell_m2 - spacetime.ingoing_mass(v)
        mask = increment > 1e3
        design = numpy.vstack(
            [numpy.ones(int(mask.sum())), v[mask]]
        ).T
        target = numpy.log(increment[mask]) + power * numpy.log(v[mask])
        coefficients, *_ = numpy.linalg.lstsq(
            design, target, rcond=None
        )
        assert abs(coefficients[1] - kappa) / kappa < 0.05

    def test_mass_inflation_diverges(self):
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        increment = spacetime._shell_m2 - spacetime.ingoing_mass(
            spacetime._shell_v
        )
        assert float(increment[-1]) > 1e5
        assert float(increment[0]) < 1.0

    def test_misner_sharp_mass_rises_across_shell(self):
        """At fixed advanced time the quasilocal mass jumps from the
        exterior value to the inflated value across the shell."""
        spacetime = ChargedVaidyaSpacetime(charge=0.9)
        # Sample where the inflation has ramped up: the advanced time
        # at which the excess mass reaches 1e4.
        increment = spacetime._shell_m2 - spacetime.ingoing_mass(
            spacetime._shell_v
        )
        index = int(numpy.searchsorted(increment, 1e4))
        v = float(spacetime._shell_v[index])
        shell_r = float(spacetime._shell_r[index])
        outside = float(
            spacetime.misner_sharp_mass(
                numpy.array([v]), numpy.array([shell_r + 0.05])
            )[0]
        )
        inside = float(
            spacetime.misner_sharp_mass(
                numpy.array([v]), numpy.array([shell_r - 0.002])
            )[0]
        )
        assert inside > 100.0 * outside


class TestFlythroughRendering:
    def test_worldline_and_frame(self):
        """A camera worldline through the shell stays timelike and
        finite, and a rendered frame is finite with visible sky."""
        from blackhorizon.frames import metric_dot
        from blackhorizon.offline.inflation import (
            build_worldline,
            camera_four_velocity,
            render_layer_frame,
        )
        from blackhorizon.offline.render import OfflineSettings

        spacetime = ChargedVaidyaSpacetime(
            charge=0.9, shell_start_v=6.0, shell_start_radius=1.2
        )
        taus, states = build_worldline(
            spacetime, 1.3, 6.0, spacetime.inner_horizon_radius * 1.001
        )
        radii = numpy.linalg.norm(states[:, 1:4], axis=-1)
        assert bool(numpy.all(numpy.diff(radii) < 1e-6))
        assert radii[-1] < spacetime.outer_horizon_radius
        assert bool(numpy.isfinite(states).all())
        # The camera 4-velocity is timelike on the frozen slice.
        for index in (0, len(states) // 2, -2):
            velocity = camera_four_velocity(
                spacetime, states[index][None, :]
            )
            slice_geo = spacetime.frozen(float(states[index][0]))
            norm = float(
                metric_dot(
                    slice_geo,
                    states[index][None, 1:4],
                    velocity[None, :],
                    velocity[None, :],
                )[0]
            )
            assert abs(norm + 1.0) < 1e-6

        settings = OfflineSettings(
            spin=0.0, supersample=1, fov_degrees=90.0, disk_enabled=False
        )
        image = render_layer_frame(
            spacetime, states[len(states) // 3], 48, 32, settings, "side"
        )
        assert bool(numpy.isfinite(image).all())
        assert float((image.max(axis=-1) > 0.01).mean()) > 0.02
