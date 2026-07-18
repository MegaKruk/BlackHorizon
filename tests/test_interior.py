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
