"""Emission physics tests: blackbody color, disk profile, redshift."""

import numpy

from blackhorizon.emission.blackbody import blackbody_lut, blackbody_rgb
from blackhorizon.emission.novikov_thorne import (
    disk_inner_radius,
    page_thorne_flux,
    peak_temperature_radius,
    temperature_lut,
    temperature_profile,
)
from blackhorizon.emission.redshift import (
    redshift_factor,
    static_observer_lapse,
)
from blackhorizon.geodesics import build_state, null_momentum_from_velocity
from blackhorizon.kerr import KerrSpacetime
from blackhorizon.realtime.reference import trace_like_shader
from blackhorizon.realtime.settings import QualityPreset, RenderSettings
from blackhorizon.tracer import RayStatus


class TestBlackbody:
    def test_temperature_color_progression(self):
        cool = blackbody_rgb(2000.0)
        neutral = blackbody_rgb(6500.0)
        hot = blackbody_rgb(20000.0)
        assert cool[0] > 2.0 * cool[2], "2000 K must be strongly red"
        assert abs(neutral[0] - neutral[2]) < 0.15, "6500 K near white"
        assert hot[2] > 2.0 * hot[0], "20000 K must be strongly blue"

    def test_lut_shape_and_range(self):
        table, log_min, log_max = blackbody_lut(size=64)
        assert table.shape == (64, 3)
        assert log_min < log_max
        assert float(table.max()) <= 1.0 + 1e-6
        assert float(table.min()) >= 0.0
        # Every entry is normalized to a unit peak channel.
        numpy.testing.assert_allclose(table.max(axis=1), 1.0, atol=1e-5)


class TestNovikovThorne:
    def test_flux_vanishes_at_isco_and_positive_outside(self):
        for spin in (0.0, 0.9, 0.998):
            st = KerrSpacetime(spin=spin)
            isco = st.isco_radius(prograde=True)
            inside = page_thorne_flux(numpy.array([isco * 0.9]), spin)
            near = page_thorne_flux(numpy.array([isco * 1.001]), spin)
            outside = page_thorne_flux(numpy.array([isco * 2.0]), spin)
            assert inside[0] == 0.0
            assert near[0] < outside[0]
            assert outside[0] > 0.0

    def test_large_radius_temperature_slope(self):
        """T approaches the Shakura-Sunyaev r^(-3/4) law far out."""
        radii = numpy.array([300.0, 600.0])
        t = temperature_profile(radii, 0.9)
        slope = numpy.log(t[1] / t[0]) / numpy.log(2.0)
        assert abs(slope + 0.75) < 0.05

    def test_peak_moves_inward_with_spin(self):
        assert peak_temperature_radius(0.9) < peak_temperature_radius(0.0)
        # Schwarzschild peak sits near the classic 9.55 M.
        assert abs(peak_temperature_radius(0.0) - 9.55) < 0.2

    def test_full_spin_range_produces_valid_luts(self):
        """Every slider-reachable spin yields a finite normalized LUT.

        Regression test: negative spins once fed the prograde ISCO to
        the signed-a Page-Thorne form, zeroing the profile and crashing
        the real-time app when the spin slider moved.
        """
        for spin in numpy.arange(-0.998, 0.999, 0.037):
            spin = float(spin)
            inner = disk_inner_radius(KerrSpacetime(spin=spin))
            table, r_in, _ = temperature_lut(spin, max(18.0, inner + 1.0))
            assert bool(numpy.isfinite(table).all()), f"spin {spin}"
            assert abs(float(table.max()) - 1.0) < 1e-6, f"spin {spin}"
            assert abs(r_in - inner) < 1e-12

    def test_retrograde_disk_less_efficient(self):
        """Counter-rotating disks truncate farther out and peak
        farther out: the classic spin-efficiency ordering."""
        st_retro = KerrSpacetime(spin=-0.9)
        st_pro = KerrSpacetime(spin=0.9)
        assert disk_inner_radius(st_retro) > 6.0 > disk_inner_radius(st_pro)
        assert (
            peak_temperature_radius(-0.9)
            > peak_temperature_radius(0.0)
            > peak_temperature_radius(0.9)
        )

    def test_lut_normalized(self):
        table, r_in, r_out = temperature_lut(0.9, 18.0, size=128)
        assert table.shape == (128,)
        assert abs(float(table.max()) - 1.0) < 1e-6
        assert table[0] < 0.05, "LUT starts at the near-zero ISCO edge"
        assert r_in < r_out


class TestRedshift:
    def _face_on_disk_hits(self, spin: float):
        """Trace a polar camera and return disk hits with momenta."""
        st = KerrSpacetime(spin=spin)
        n = 40
        # Aim a fan from the pole toward disk radii between 4 and 15 M.
        aims = numpy.linspace(0.004, 0.015, n)
        distance = 1000.0
        positions = numpy.tile([[0.0, 0.0, distance]], (n, 1))
        directions = numpy.stack(
            [aims, numpy.zeros(n), -numpy.ones(n)], axis=-1
        )
        directions /= numpy.linalg.norm(directions, axis=-1, keepdims=True)
        momenta = null_momentum_from_velocity(
            st, positions, directions, time_orientation="past"
        )
        state0 = build_state(positions, momenta)
        settings = RenderSettings(spin=spin).apply_preset(QualityPreset.HIGH)
        result = trace_like_shader(
            st,
            state0,
            settings,
            escape_radius=1.2 * distance,
            disk_radii=(st.isco_radius(True), 30.0),
        )
        hits = result.status == int(RayStatus.DISK)
        return st, result, hits

    def test_face_on_schwarzschild_matches_analytic(self):
        """Face-on g equals sqrt(1 - 3M/r), the classic exact result."""
        st, result, hits = self._face_on_disk_hits(0.0)
        assert int(hits.sum()) >= 10
        momenta = numpy.concatenate(
            [numpy.ones((int(hits.sum()), 1)), result.hit_momenta[hits][:, 1:4]],
            axis=1,
        )
        g = redshift_factor(st, result.hit_positions[hits], momenta)
        expected = numpy.sqrt(1.0 - 3.0 / result.hit_radii[hits])
        numpy.testing.assert_allclose(g, expected, rtol=2e-3)

    def test_edge_on_doppler_asymmetry(self):
        """Approaching-side photons are blueshifted relative to receding."""
        st = KerrSpacetime(spin=0.0)
        distance = 1000.0
        aims = numpy.array([0.010, -0.010])
        # Start slightly above the plane and descend so the rays cross
        # the equator near the hole, on opposite azimuthal sides.
        positions = numpy.tile([[distance, 0.0, 30.0]], (2, 1))
        directions = numpy.stack(
            [-numpy.ones(2), aims, numpy.full(2, -0.0302)], axis=-1
        )
        directions /= numpy.linalg.norm(directions, axis=-1, keepdims=True)
        momenta = null_momentum_from_velocity(
            st, positions, directions, time_orientation="past"
        )
        state0 = build_state(positions, momenta)
        settings = RenderSettings(spin=0.0).apply_preset(QualityPreset.HIGH)
        result = trace_like_shader(
            st, state0, settings, escape_radius=1.2 * distance,
            disk_radii=(st.isco_radius(True), 30.0),
        )
        assert bool(
            numpy.all(result.status == int(RayStatus.DISK))
        ), "both probe rays must hit the disk"
        momenta4 = numpy.concatenate(
            [numpy.ones((2, 1)), result.hit_momenta[:, 1:4]], axis=1
        )
        g = redshift_factor(st, result.hit_positions, momenta4)
        assert abs(g[0] - g[1]) > 0.2, "Doppler split must be strong"

    def test_static_observer_lapse(self):
        st = KerrSpacetime(spin=0.0)
        lapse = static_observer_lapse(st, numpy.array([10.0, 0.0, 0.0]))
        assert abs(lapse - 1.0 / numpy.sqrt(1.0 - 0.2)) < 1e-12
