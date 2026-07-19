"""Interior physics tests for the Stage 5 doomed-observer machinery.

Anchored to analytic results: the rain observer's factor-2 redshift of
overhead starlight at the horizon, the pi M maximal interior lifetime,
the impossibility of increasing r inside, and the corrected sky
visibility (more than half the sky shows the outside universe just
inside the horizon; NASA Schnittman and Powell 2024, Hamilton and
Polhemus arXiv:0903.4717).
"""

import numpy

from blackhorizon.frames import (
    build_tetrad,
    rain_four_velocity,
    raise_index,
    tetrad_ray_momenta,
)
from blackhorizon.geodesics import build_state, geodesic_rhs, hamiltonian
from blackhorizon.integrators import rk4_step
from blackhorizon.kerr import KerrSpacetime
from blackhorizon.realtime.reference import trace_like_shader
from blackhorizon.realtime.settings import QualityPreset, RenderSettings
from blackhorizon.tracer import RayStatus


def fibonacci_sphere(n: int) -> numpy.ndarray:
    """Nearly uniform unit directions on the sphere, shape (n, 3)."""
    indices = numpy.arange(n, dtype=float) + 0.5
    phi = numpy.arccos(1.0 - 2.0 * indices / n)
    theta = numpy.pi * (1.0 + 5.0**0.5) * indices
    return numpy.stack(
        [
            numpy.cos(theta) * numpy.sin(phi),
            numpy.sin(theta) * numpy.sin(phi),
            numpy.cos(phi),
        ],
        axis=-1,
    )


def _interior_sky_fraction(radius: float, n_rays: int = 400) -> float:
    """Fraction of the local sky showing the outside universe."""
    st = KerrSpacetime(spin=0.0)
    pos = numpy.array([radius, 0.0, 0.0])
    u = rain_four_velocity(st, pos[None, :])[0]
    tetrad = build_tetrad(
        st, pos, u, numpy.array([-1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 1.0])
    )
    momenta = tetrad_ray_momenta(
        st, pos, tetrad, fibonacci_sphere(n_rays)
    )
    state0 = build_state(numpy.tile(pos[None, :], (n_rays, 1)), momenta)
    settings = RenderSettings(spin=0.0, disk_enabled=False).apply_preset(
        QualityPreset.HIGH
    )
    result = trace_like_shader(
        st, state0, settings, escape_radius=400.0, interior_stop=0.02
    )
    return float(numpy.mean(result.status == int(RayStatus.ESCAPED)))


class TestInteriorView:
    def test_more_than_half_sky_visible_just_inside(self):
        """Just inside the horizon the outside universe still fills
        more than half of the rain observer's sky."""
        fraction = _interior_sky_fraction(1.9)
        assert fraction > 0.5, f"sky fraction {fraction}"

    def test_sky_narrows_approaching_singularity(self):
        deep = _interior_sky_fraction(0.5)
        shallow = _interior_sky_fraction(1.9)
        assert deep < shallow
        assert deep > 0.05, "some sky remains visible until very late"

    def test_all_escaping_rays_have_finite_shift(self):
        """No infinite blueshift at the event horizon: conserved ray
        energies (hence sky shift factors) stay finite and moderate."""
        st = KerrSpacetime(spin=0.0)
        pos = numpy.array([1.95, 0.0, 0.0])
        u = rain_four_velocity(st, pos[None, :])[0]
        tetrad = build_tetrad(
            st,
            pos,
            u,
            numpy.array([-1.0, 0.0, 0.0]),
            numpy.array([0.0, 0.0, 1.0]),
        )
        directions = fibonacci_sphere(200)
        momenta = tetrad_ray_momenta(st, pos, tetrad, directions)
        state0 = build_state(
            numpy.tile(pos[None, :], (200, 1)), momenta
        )
        settings = RenderSettings(
            spin=0.0, disk_enabled=False
        ).apply_preset(QualityPreset.HIGH)
        result = trace_like_shader(
            st, state0, settings, escape_radius=400.0, interior_stop=0.02
        )
        escaped = result.status == int(RayStatus.ESCAPED)
        assert int(escaped.sum()) > 100
        energies = momenta[escaped, 0]
        assert bool(numpy.all(energies > 0.0))
        assert float(energies.max()) < 50.0

    def test_rain_horizon_overhead_redshift_factor_two(self):
        """A rain observer at the horizon sees light falling from
        directly overhead redshifted by exactly a factor of two:
        g = 1 / (1 + sqrt(2M/r)) = 1/2 at r = 2M."""
        st = KerrSpacetime(spin=0.0)
        for radius, expected in ((2.0, 0.5), (8.0, 1.0 / 1.5)):
            pos = numpy.array([radius, 0.0, 0.0])
            u = rain_four_velocity(st, pos[None, :])[0]
            tetrad = build_tetrad(
                st,
                pos,
                u,
                numpy.array([1.0, 0.0, 0.0]),
                numpy.array([0.0, 0.0, 1.0]),
            )
            # Looking radially outward: the ray's conserved energy is
            # the sky frequency; camera frequency is one by
            # construction, so g = 1 / E.
            momenta = tetrad_ray_momenta(
                st, pos, tetrad, numpy.array([[1.0, 0.0, 0.0]])
            )
            g = 1.0 / float(momenta[0, 0])
            assert abs(g - expected) < 1e-12


class TestDoomedObserver:
    def _radial_interior_state(self, radius: float, w: float):
        """Timelike state at (radius, 0, 0) with covariant p = (p_t, w r_hat)."""
        st = KerrSpacetime(spin=0.0)
        pos = numpy.array([[radius, 0.0, 0.0]])
        return st, pos

    def test_maximal_interior_lifetime_pi_m(self):
        """The E = 0 interior geodesic realizes tau_max = pi M.

        Integrated proper time from just inside the horizon to just
        outside the center matches the analytic
        tau = integral dr / sqrt(2M/r - 1) to 0.1 percent, and the
        full-range value is pi M (Toporensky and Zaslavskii,
        arXiv:1905.02150; Lewis and Kwan 2007).
        """
        st = KerrSpacetime(spin=0.0)
        r0, r_stop = 1.98, 0.02
        pos = numpy.array([[r0, 0.0, 0.0]])
        h_field = st.geometry(pos[:, 0], pos[:, 1], pos[:, 2]).h
        w = 1.0 / numpy.sqrt(2.0 * float(h_field[0]) - 1.0)
        momentum = numpy.array([[0.0, w, 0.0, 0.0]])
        state = build_state(pos, momentum)
        assert abs(float(hamiltonian(st, state)[0]) + 0.5) < 1e-12
        velocity = raise_index(st, pos, momentum)
        assert float(velocity[0, 1]) < 0.0, "future-directed infall"

        def rhs(batch):
            return geodesic_rhs(st, batch)

        h = numpy.array([2e-4])
        tau = 0.0
        for _ in range(200000):
            state = rk4_step(rhs, state, h)
            tau += float(h[0])
            radius = float(
                st.kerr_schild_radius(state[:, 1], state[:, 2], state[:, 3])[0]
            )
            if radius <= r_stop:
                break
        r_grid = numpy.linspace(r_stop, r0, 400000)
        analytic = numpy.trapezoid(
            1.0 / numpy.sqrt(2.0 / r_grid - 1.0), r_grid
        )
        assert abs(tau - analytic) / analytic < 1e-3
        # Full-range value via r = 2 sin^2(theta), which removes the
        # inverse-square-root endpoint singularity: the integrand
        # becomes 4 sin^2(theta) and the integral is exactly pi M.
        theta = numpy.linspace(0.0, numpy.pi / 2.0, 200000)
        pi_m = numpy.trapezoid(4.0 * numpy.sin(theta) ** 2, theta)
        assert abs(pi_m - numpy.pi) < 1e-9

    def test_radius_never_increases_inside(self):
        """Even launched with maximal outward spatial momentum, an
        interior worldline has monotonically decreasing radius."""
        st = KerrSpacetime(spin=0.0)
        pos = numpy.array([[1.7, 0.0, 0.0]])
        # Outward-directed timelike momentum: covariant p = (p_t, w r_hat)
        # with w > 0 as large as the mass shell allows for chosen p_t.
        p_t = -0.2
        h_field = float(st.geometry(pos[:, 0], pos[:, 1], pos[:, 2]).h[0])
        # Solve -p_t^2 + w^2 - 2H(-p_t + w)^2 = -1 for the larger root.
        a_c = 1.0 - 2.0 * h_field
        b_c = 4.0 * h_field * p_t
        c_c = -p_t**2 - 2.0 * h_field * p_t**2 + 1.0
        roots = numpy.roots([a_c, b_c, c_c])
        momentum = None
        for w in sorted(roots.real, reverse=True):
            candidate = numpy.array([[p_t, float(w), 0.0, 0.0]])
            if abs(float(hamiltonian(st, build_state(pos, candidate))[0]) + 0.5) > 1e-9:
                continue
            velocity = raise_index(st, pos, candidate)
            if float(velocity[0, 0]) > 0.0:
                momentum = candidate
                break
        assert momentum is not None, "no future-directed root found"
        state = build_state(pos, momentum)

        def rhs(batch):
            return geodesic_rhs(st, batch)

        h = numpy.array([1e-3])
        radii = []
        for _ in range(20000):
            state = rk4_step(rhs, state, h)
            radii.append(
                float(
                    st.kerr_schild_radius(
                        state[:, 1], state[:, 2], state[:, 3]
                    )[0]
                )
            )
            if radii[-1] < 0.05:
                break
        radii = numpy.asarray(radii)
        assert radii[-1] < 0.06, "worldline must reach the center"
        assert bool(numpy.all(numpy.diff(radii) < 1e-9)), (
            "radius must never increase inside the horizon"
        )

    def test_rain_faster_than_maximal(self):
        """The rain (E = 1) plunge reaches the center in less proper
        time than the maximal E = 0 trajectory from the same radius."""
        st = KerrSpacetime(spin=0.0)
        r0, r_stop = 1.9, 0.05

        def lifetime(momentum):
            state = build_state(
                numpy.array([[r0, 0.0, 0.0]]), momentum
            )

            def rhs(batch):
                return geodesic_rhs(st, batch)

            h = numpy.array([5e-4])
            tau = 0.0
            for _ in range(100000):
                state = rk4_step(rhs, state, h)
                tau += float(h[0])
                radius = float(
                    st.kerr_schild_radius(
                        state[:, 1], state[:, 2], state[:, 3]
                    )[0]
                )
                if radius <= r_stop:
                    return tau
            raise AssertionError("worldline did not reach the center")

        pos = numpy.array([[r0, 0.0, 0.0]])
        h_field = float(st.geometry(pos[:, 0], pos[:, 1], pos[:, 2]).h[0])
        w_max = 1.0 / numpy.sqrt(2.0 * h_field - 1.0)
        tau_maximal = lifetime(numpy.array([[0.0, w_max, 0.0, 0.0]]))

        root = numpy.sqrt(2.0 * h_field)
        w_rain = -root / (1.0 + root)
        tau_rain = lifetime(numpy.array([[-1.0, w_rain, 0.0, 0.0]]))
        assert tau_rain < tau_maximal


class TestKerrInteriorTermination:
    """Spinning holes: the journey ends at the Cauchy horizon.

    Regression tests for a field-reported crash: with spin 0.9 the
    worldline formerly penetrated the inner horizon, where r turns
    spacelike again and the state left the mass shell, making the
    camera 4-velocity non-timelike and crashing tetrad construction.
    Realistically the blue sheet at r_minus ends the infall
    (Poisson-Israel mass inflation; Dafermos-Luk arXiv:1710.01722).
    """

    def test_worldline_terminates_at_cauchy_horizon(self):
        from blackhorizon.realtime.infall import InfallState

        spacetime = KerrSpacetime(spin=0.9)
        position = numpy.array(
            [spacetime.outer_horizon_radius * 0.999, 0.1, 0.05]
        )
        infall = InfallState.from_crossing(spacetime, position, 0.02)
        assert (
            infall.stop_radius
            >= spacetime.inner_horizon_radius * 1.01
        )
        forward = numpy.array([1.0, 0.0, 0.0])
        up = numpy.array([0.0, 0.0, 1.0])
        for _ in range(3000):
            if infall.terminated():
                break
            infall.advance(0.016 * 0.2)
            # The user-facing crash path: tetrad construction from the
            # camera state must succeed on every frame.
            build_tetrad(
                spacetime, infall.position, infall.four_velocity,
                forward, up,
            )
        assert infall.terminated()
        assert infall.radius >= spacetime.inner_horizon_radius
        assert bool(numpy.isfinite(infall.state).all())
        # Post-termination frames keep rendering (the app does not stop).
        for _ in range(5):
            infall.advance(0.01)
            build_tetrad(
                spacetime, infall.position, infall.four_velocity,
                forward, up,
            )

    def test_thrust_spam_through_kerr_plunge(self):
        """Mashing the thrusters all the way down must stay stable."""
        from blackhorizon.realtime.infall import InfallState

        spacetime = KerrSpacetime(spin=0.9)
        position = numpy.array(
            [spacetime.outer_horizon_radius * 0.998, 0.05, 0.02]
        )
        infall = InfallState.from_crossing(spacetime, position, 0.02)
        forward = numpy.array([0.3, 0.9, 0.1])
        forward /= numpy.linalg.norm(forward)
        up = numpy.array([0.0, 0.0, 1.0])
        for _ in range(3000):
            if infall.terminated():
                break
            infall.advance(0.016 * 0.2)
            infall.thrust(
                numpy.array([1.0, 0.3, 0.0]), 0.8 * 0.016, forward, up,
                recompute_lookahead=False,
            )
        assert infall.terminated()
        assert bool(numpy.isfinite(infall.state).all())

    def test_schwarzschild_stop_unchanged(self):
        from blackhorizon.realtime.infall import InfallState

        spacetime = KerrSpacetime(spin=0.0)
        infall = InfallState.from_crossing(
            spacetime, numpy.array([1.99, 0.0, 0.0]), 0.02
        )
        assert abs(infall.stop_radius - 0.02) < 1e-12


class TestIdealizedJourney:
    """The labeled idealized continuation into exact vacuum Kerr.

    The single stationary ingoing Kerr-Schild chart covers the inward
    crossing of both horizons; worldlines that reach the ring plane
    (the gateway to negative r) or turn outward and asymptote the
    Cauchy horizon leave the chart and terminate with reason "chart".
    """

    def test_rain_continues_past_cauchy_horizon(self):
        from blackhorizon.realtime.infall import InfallState

        spacetime = KerrSpacetime(spin=0.9)
        r_minus = spacetime.inner_horizon_radius
        position = numpy.array(
            [spacetime.outer_horizon_radius * 0.999, 0.1, 0.05]
        )
        infall = InfallState.from_crossing(
            spacetime, position, 0.02, journey="idealized"
        )
        assert abs(infall.stop_radius - 0.02) < 1e-12
        crossed = False
        for _ in range(5000):
            if infall.terminated():
                break
            infall.advance(0.016 * 0.2)
            if infall.radius < r_minus:
                crossed = True
                # The mass shell holds through the Cauchy horizon.
                value = float(
                    hamiltonian(spacetime, infall.state)[0]
                )
                assert abs(value + 0.5) < 1e-6
        assert crossed, "must cross the Cauchy horizon inward"
        assert infall.terminated()
        assert infall.radius < r_minus
        assert bool(numpy.isfinite(infall.state).all())

    def test_outward_branch_ends_at_chart_boundary(self):
        """Inside r_minus the radius may legally increase; the branch
        trying to exit into the next universe leaves the chart."""
        import math

        from blackhorizon.frames import lower_index
        from blackhorizon.realtime.infall import InfallState

        spacetime = KerrSpacetime(spin=0.9)
        position = numpy.array([[0.35, 0.0, 0.1]])
        u_rain = rain_four_velocity(spacetime, position)
        tetrad = build_tetrad(
            spacetime,
            position[0],
            u_rain[0],
            numpy.array([1.0, 0.0, 0.0]),
            numpy.array([0.0, 0.0, 1.0]),
        )
        boosted = (
            math.cosh(1.2) * tetrad[0] + math.sinh(1.2) * tetrad[1]
        )
        momentum = lower_index(spacetime, position, boosted[None, :])
        infall = InfallState(
            spacetime,
            build_state(position, momentum),
            0.02,
            journey="idealized",
        )
        radii = [infall.radius]
        for _ in range(2000):
            if infall.terminated():
                break
            infall.advance(0.005)
            radii.append(infall.radius)
        assert max(radii) > radii[0] + 0.1, (
            "radius must legally increase inside the Cauchy horizon"
        )
        assert infall.terminated()
        assert infall.termination_reason == "chart"
        assert bool(numpy.isfinite(infall.state).all())

    def test_camera_inside_cauchy_horizon_sees_sky(self):
        spacetime = KerrSpacetime(spin=0.9)
        position = numpy.array([0.4, 0.05, 0.1])
        u = rain_four_velocity(spacetime, position[None, :])[0]
        tetrad = build_tetrad(
            spacetime,
            position,
            u,
            numpy.array([1.0, 0.0, 0.0]),
            numpy.array([0.0, 0.0, 1.0]),
        )
        momenta = tetrad_ray_momenta(
            spacetime, position, tetrad, fibonacci_sphere(300)
        )
        state0 = build_state(
            numpy.tile(position[None, :], (300, 1)), momenta
        )
        settings = RenderSettings(
            spin=0.9,
            interior_mode=True,
            interior_journey="idealized",
            disk_enabled=False,
        ).apply_preset(QualityPreset.HIGH)
        result = trace_like_shader(
            spacetime,
            state0,
            settings,
            escape_radius=400.0,
            interior_stop=0.02,
        )
        escaped = float(
            (result.status == int(RayStatus.ESCAPED)).mean()
        )
        assert escaped > 0.5, (
            "the outside universe is visible through the Cauchy "
            f"horizon (escaped {escaped})"
        )

    def test_live_journey_switch_updates_terminal_surface(self):
        from blackhorizon.realtime.infall import InfallState

        spacetime = KerrSpacetime(spin=0.9)
        position = numpy.array(
            [spacetime.outer_horizon_radius * 0.99, 0.0, 0.05]
        )
        infall = InfallState.from_crossing(
            spacetime, position, 0.02, journey="realistic"
        )
        realistic_stop = infall.stop_radius
        assert realistic_stop >= spacetime.inner_horizon_radius
        infall.set_journey("idealized")
        assert abs(infall.stop_radius - 0.02) < 1e-12
        infall.set_journey("realistic")
        assert abs(infall.stop_radius - realistic_stop) < 1e-12
