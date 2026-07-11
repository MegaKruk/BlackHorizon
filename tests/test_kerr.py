"""Geometry tests: analytic radii, Kerr-Schild identities, metric algebra."""

import math

import numpy
import pytest

from blackhorizon.kerr import KerrSpacetime


def random_points(n: int, r_min: float = 2.5, r_max: float = 40.0, seed: int = 3):
    """Random points outside the strong-field guard radius."""
    rng = numpy.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= numpy.linalg.norm(u, axis=-1, keepdims=True)
    radii = rng.uniform(r_min, r_max, size=(n, 1))
    return radii * u


class TestAnalyticRadii:
    def test_horizons(self):
        assert KerrSpacetime(spin=0.0).outer_horizon_radius == pytest.approx(2.0)
        assert KerrSpacetime(spin=1.0).outer_horizon_radius == pytest.approx(1.0)
        st = KerrSpacetime(spin=0.6)
        assert st.outer_horizon_radius == pytest.approx(1.0 + 0.8)
        assert st.inner_horizon_radius == pytest.approx(1.0 - 0.8)

    def test_isco_limits(self):
        assert KerrSpacetime(spin=0.0).isco_radius() == pytest.approx(6.0)
        assert KerrSpacetime(spin=1.0).isco_radius(prograde=True) == pytest.approx(1.0)
        assert KerrSpacetime(spin=1.0).isco_radius(prograde=False) == pytest.approx(9.0)

    def test_photon_orbit_limits(self):
        assert KerrSpacetime(spin=0.0).photon_orbit_radius() == pytest.approx(3.0)
        assert KerrSpacetime(spin=1.0).photon_orbit_radius(True) == pytest.approx(1.0)
        assert KerrSpacetime(spin=1.0).photon_orbit_radius(False) == pytest.approx(4.0)

    def test_ergosphere(self):
        st = KerrSpacetime(spin=0.9)
        # On the axis the ergosphere touches the horizon.
        assert st.ergosphere_radius(0.0) == pytest.approx(st.outer_horizon_radius)
        # In the equatorial plane it sits at r = 2M.
        assert st.ergosphere_radius(math.pi / 2.0) == pytest.approx(2.0)

    def test_kerr_bound_enforced(self):
        with pytest.raises(ValueError):
            KerrSpacetime(spin=1.01)
        with pytest.raises(ValueError):
            KerrSpacetime(mass=-1.0)


class TestKerrSchildRadius:
    def test_on_axis(self):
        st = KerrSpacetime(spin=0.8)
        z = numpy.array([1.5, 3.0, -7.0])
        r = st.kerr_schild_radius(numpy.zeros(3), numpy.zeros(3), z)
        numpy.testing.assert_allclose(r, numpy.abs(z), rtol=1e-14)

    def test_equatorial_plane(self):
        a = 0.8
        st = KerrSpacetime(spin=a)
        x = numpy.array([2.0, 5.0, 11.0])
        r = st.kerr_schild_radius(x, numpy.zeros(3), numpy.zeros(3))
        numpy.testing.assert_allclose(r, numpy.sqrt(x * x - a * a), rtol=1e-14)

    def test_reduces_to_spherical_radius_for_zero_spin(self):
        st = KerrSpacetime(spin=0.0)
        pts = random_points(50)
        r = st.kerr_schild_radius(pts[:, 0], pts[:, 1], pts[:, 2])
        numpy.testing.assert_allclose(
            r, numpy.linalg.norm(pts, axis=-1), rtol=1e-13
        )


class TestMetricAlgebra:
    def test_null_covector_in_flat_metric(self):
        st = KerrSpacetime(spin=0.95)
        pts = random_points(200)
        geo = st.geometry(pts[:, 0], pts[:, 1], pts[:, 2])
        # eta^ab l_a l_b = -1 + |l_spatial|^2 must vanish.
        norm = -1.0 + numpy.sum(geo.l * geo.l, axis=-1)
        numpy.testing.assert_allclose(norm, 0.0, atol=1e-12)

    def test_inverse_metric_identity(self):
        st = KerrSpacetime(spin=0.95)
        pts = random_points(100)
        positions = numpy.concatenate(
            [numpy.zeros((100, 1)), pts], axis=-1
        )
        g = st.metric(positions)
        g_inv = st.inverse_metric(positions)
        identity = numpy.einsum("nab,nbc->nac", g, g_inv)
        expected = numpy.broadcast_to(numpy.eye(4), identity.shape)
        numpy.testing.assert_allclose(identity, expected, atol=1e-12)

    def test_schwarzschild_limit(self):
        st = KerrSpacetime(spin=0.0)
        pts = random_points(100)
        r = numpy.linalg.norm(pts, axis=-1)
        geo = st.geometry(pts[:, 0], pts[:, 1], pts[:, 2])
        numpy.testing.assert_allclose(geo.h, 1.0 / r, rtol=1e-13)
        numpy.testing.assert_allclose(
            geo.l, pts / r[:, None], rtol=1e-13, atol=1e-15
        )
