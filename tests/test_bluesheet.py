"""Blue-sheet physics and rendering integration (Stage 6).

The amplification law B = x_match / x follows from steady external
illumination entering at all advanced times, received near the Cauchy
horizon amplified by exp(kappa_minus v), with the near-horizon
relation v = -(1/kappa_minus) ln x along infalling worldlines
(Poisson-Israel 1990; Ori 1991; Hamilton-Avelino arXiv:0811.1926).
Exact-Kerr geometry carries the angular structure; the law is purely
multiplicative on the observer lapse.
"""

import numpy

from blackhorizon.emission.bluesheet import (
    DISPLAY_CAP,
    blueshift_amplification,
    display_amplification,
    inner_surface_gravity,
    proximity,
    sheet_radiance,
    whiteout_fraction,
)
from blackhorizon.kerr import KerrSpacetime


class TestBlueSheetLaw:
    def test_inner_surface_gravity_analytic(self):
        spacetime = KerrSpacetime(spin=0.9)
        r_plus = spacetime.outer_horizon_radius
        r_minus = spacetime.inner_horizon_radius
        expected = (r_plus - r_minus) / (2.0 * (r_minus**2 + 0.81))
        assert abs(inner_surface_gravity(spacetime) - expected) < 1e-14
        assert inner_surface_gravity(KerrSpacetime(spin=0.0)) == 0.0

    def test_amplification_monotone_and_continuous(self):
        spacetime = KerrSpacetime(spin=0.9)
        r_plus = spacetime.outer_horizon_radius
        r_minus = spacetime.inner_horizon_radius
        radii = numpy.linspace(r_minus * 1.001, r_plus * 0.99, 400)
        values = blueshift_amplification(spacetime, radii)
        assert bool(numpy.all(numpy.diff(values) <= 1e-12))
        assert float(values[-1]) == 1.0
        assert float(values[0]) > 30.0
        # Continuity at the matching proximity.
        r_match = r_minus + 0.5 * (r_plus - r_minus)
        eps = 1e-9
        above = float(
            blueshift_amplification(
                spacetime, numpy.array([r_match + eps])
            )[0]
        )
        below = float(
            blueshift_amplification(
                spacetime, numpy.array([r_match - eps])
            )[0]
        )
        assert abs(above - below) < 1e-6

    def test_schwarzschild_and_idealized_honesty(self):
        """No inner horizon, no radiation model: identity everywhere."""
        spacetime = KerrSpacetime(spin=0.0)
        radii = numpy.array([1.9, 1.0, 0.1, 0.02])
        assert bool(
            numpy.all(
                blueshift_amplification(spacetime, radii) == 1.0
            )
        )
        assert bool(
            numpy.all(proximity(spacetime, radii) == 1.0)
        )

    def test_display_adaptation_and_whiteout(self):
        values = numpy.array([1.0, 5.0, 8.0, 20.0, 60.0])
        display = display_amplification(values)
        assert float(display.max()) == DISPLAY_CAP
        assert bool(numpy.all(numpy.diff(display) >= 0.0))
        white = whiteout_fraction(values)
        assert float(white[0]) == 0.0
        assert float(white[2]) == 0.0
        assert 0.0 < float(white[3]) < 0.92
        assert abs(float(white[4]) - 0.92) < 1e-9
        assert float(sheet_radiance(numpy.array([1.0]))[0]) == 0.0
        assert float(sheet_radiance(numpy.array([8.0]))[0]) > 10.0


class TestBlueSheetRendering:
    def test_offline_flare_brightens_on_approach(self):
        """A realistic Kerr interior frame brightens and blue-shifts
        as the camera closes on the inner horizon; the idealized
        counterpart at the same position stays dark."""
        from blackhorizon.frames import build_tetrad, rain_four_velocity
        from blackhorizon.offline.render import OfflineSettings, render_hdr
        from blackhorizon.realtime.fly_camera import FlyCamera

        spacetime = KerrSpacetime(spin=0.9)
        r_minus = spacetime.inner_horizon_radius
        settings = OfflineSettings(
            spin=0.9, supersample=1, fov_degrees=80.0, disk_enabled=False
        )

        def frame(radius, stop):
            position = numpy.array([0.03, 0.03, radius])
            velocity = rain_four_velocity(
                spacetime, position[None, :]
            )[0]
            tetrad = build_tetrad(
                spacetime,
                position,
                velocity,
                numpy.array([0.0, 1.0, 0.0]),
                numpy.array([0.0, 0.0, 1.0]),
            )
            camera = FlyCamera(
                position=position.copy(), yaw=0.0, pitch=0.0
            )
            return render_hdr(
                camera, 48, 32, settings, progress=False,
                camera_tetrad=tetrad, interior_stop=stop,
            )

        realistic_stop = r_minus * 1.02
        far = frame(1.2, realistic_stop)
        close = frame(0.68, realistic_stop)
        assert float(close.mean()) > 5.0 * max(float(far.mean()), 1e-6)
        # Blue dominates red in the flare.
        assert float(close[:, :, 2].mean()) > float(
            close[:, :, 0].mean()
        )
        idealized = frame(0.68, 0.02)
        assert float(idealized.mean()) < 0.2 * float(close.mean())
        assert bool(numpy.isfinite(close).all())
