"""Analytic gradients of the Kerr-Schild geometry vs finite differences."""

import numpy

from blackhorizon.kerr import KerrSpacetime

from test_kerr import random_points


def finite_difference_gradients(st: KerrSpacetime, pts: numpy.ndarray, eps: float):
    """Central-difference gradients of H and l at the given points."""
    n = pts.shape[0]
    grad_h = numpy.zeros((n, 3))
    grad_l = numpy.zeros((n, 3, 3))
    for axis in range(3):
        shift = numpy.zeros(3)
        shift[axis] = eps
        plus = pts + shift
        minus = pts - shift
        geo_p = st.geometry(plus[:, 0], plus[:, 1], plus[:, 2])
        geo_m = st.geometry(minus[:, 0], minus[:, 1], minus[:, 2])
        grad_h[:, axis] = (geo_p.h - geo_m.h) / (2.0 * eps)
        grad_l[:, axis, :] = (geo_p.l - geo_m.l) / (2.0 * eps)
    return grad_h, grad_l


def test_gradients_match_finite_differences():
    st = KerrSpacetime(spin=0.9)
    pts = random_points(300, r_min=2.2, r_max=60.0, seed=11)
    geo = st.geometry(pts[:, 0], pts[:, 1], pts[:, 2], gradients=True)
    fd_h, fd_l = finite_difference_gradients(st, pts, eps=1e-6)
    numpy.testing.assert_allclose(geo.grad_h, fd_h, rtol=1e-5, atol=1e-10)
    numpy.testing.assert_allclose(geo.grad_l, fd_l, rtol=1e-5, atol=1e-9)


def test_gradients_zero_spin():
    st = KerrSpacetime(spin=0.0)
    pts = random_points(100, seed=13)
    r = numpy.linalg.norm(pts, axis=-1)
    geo = st.geometry(pts[:, 0], pts[:, 1], pts[:, 2], gradients=True)
    # H = M / r so grad H = -M x_i / r^3.
    expected = -pts / (r ** 3)[:, None]
    numpy.testing.assert_allclose(geo.grad_h, expected, rtol=1e-12)
