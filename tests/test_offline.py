"""Stage 4 tests: symplectic integration and the offline renderer."""

import pathlib

import numpy
import pytest

from blackhorizon.geodesics import (
    build_state,
    geodesic_rhs,
    hamiltonian,
    timelike_momentum,
)
from blackhorizon.integrators import implicit_midpoint_step, rk4_step
from blackhorizon.kerr import KerrSpacetime
from blackhorizon.offline.camera_path import (
    CameraKeyframe,
    CameraPath,
    orbit_path,
)
from blackhorizon.offline.post import (
    add_bloom,
    aces_tonemap,
    develop,
    encode_srgb,
    gaussian_blur,
)
from blackhorizon.offline.render import (
    OfflineSettings,
    render_hdr,
    subpixel_directions,
)
from blackhorizon.offline.video import downsample, frame_bloom
from blackhorizon.realtime.fly_camera import FlyCamera


class TestImplicitMidpoint:
    def _orbit_state(self):
        spacetime = KerrSpacetime(spin=0.9)
        pos = numpy.array([[14.0, 0.0, 0.0]])
        vel = numpy.array([[0.0, 0.21, 0.02]])
        return spacetime, build_state(
            pos, timelike_momentum(spacetime, pos, vel)
        )

    def test_second_order_convergence(self):
        """Halving the step reduces the one-orbit error about 4x."""
        spacetime, state0 = self._orbit_state()

        def rhs(batch):
            return geodesic_rhs(spacetime, batch)

        def endpoint(h_value, steps):
            state = state0.copy()
            h = numpy.array([h_value])
            for _ in range(steps):
                state = implicit_midpoint_step(rhs, state, h, iterations=6)
            return state

        reference = endpoint(0.025, 4000)
        coarse = endpoint(0.2, 500)
        fine = endpoint(0.1, 1000)
        error_coarse = numpy.abs(coarse[0, 1:4] - reference[0, 1:4]).max()
        error_fine = numpy.abs(fine[0, 1:4] - reference[0, 1:4]).max()
        order = numpy.log2(error_coarse / error_fine)
        assert 1.6 < order < 2.6

    def test_bounded_energy_error_where_rk4_drifts(self):
        """At large fixed steps the midpoint Hamiltonian error stays
        bounded while the RK4 error grows secularly."""
        spacetime, state0 = self._orbit_state()

        def rhs(batch):
            return geodesic_rhs(spacetime, batch)

        h = numpy.array([0.6])
        samples = {}
        for name, stepper in (
            ("rk4", rk4_step),
            ("midpoint", implicit_midpoint_step),
        ):
            state = state0.copy()
            errors = []
            for i in range(30000):
                state = stepper(rhs, state, h)
                if i % 2500 == 2499:
                    errors.append(
                        abs(float(hamiltonian(spacetime, state)[0]) + 0.5)
                    )
            samples[name] = numpy.asarray(errors)

        # RK4 truncation error accumulates secularly: every sample
        # exceeds the previous one and the final error dwarfs the first.
        rk4 = samples["rk4"]
        assert bool(numpy.all(numpy.diff(rk4) > 0.0))
        assert rk4[-1] > 5.0 * rk4[0]

        # The symplectic midpoint error oscillates and returns: the
        # final sample sits far below the peak instead of at it, and
        # the peak itself stays bounded.
        midpoint = samples["midpoint"]
        assert float(midpoint[-1]) < 0.05 * float(midpoint.max())
        assert float(midpoint.max()) < 1e-2


class TestPost:
    def test_gaussian_blur_preserves_mean(self):
        rng = numpy.random.default_rng(0)
        image = rng.random((40, 50, 3))
        blurred = gaussian_blur(image, sigma=2.0)
        assert abs(blurred.mean() - image.mean()) < 0.01
        assert blurred.std() < image.std()

    def test_bloom_only_brightens(self):
        image = numpy.zeros((41, 41, 3))
        image[20, 20] = 30.0
        bloomed = add_bloom(image, threshold=1.0, strength=0.5, sigma=3.0)
        assert bool(numpy.all(bloomed >= image - 1e-12))
        assert bloomed[20, 24].sum() > 0.0, "glow must spread outward"

    def test_bloom_ignores_dim_pixels(self):
        image = numpy.full((16, 16, 3), 0.2)
        bloomed = add_bloom(image, threshold=1.0, strength=0.5, sigma=2.0)
        numpy.testing.assert_allclose(bloomed, image, atol=1e-12)

    def test_aces_monotonic_and_bounded(self):
        values = numpy.linspace(0.0, 50.0, 500)[:, None, None] * numpy.ones(
            (1, 1, 3)
        )
        mapped = aces_tonemap(values)
        assert float(mapped.min()) >= 0.0
        assert float(mapped.max()) <= 1.0
        flat = mapped[:, 0, 0]
        assert bool(numpy.all(numpy.diff(flat) >= -1e-9))

    def test_develop_output_type(self):
        image = develop(numpy.full((8, 8, 3), 0.5), exposure=1.0)
        assert image.dtype == numpy.uint8
        assert image.shape == (8, 8, 3)

    def test_srgb_encode_endpoints(self):
        encoded = encode_srgb(numpy.array([[[0.0, 1.0, 0.5]]]))
        assert encoded[0, 0, 0] == 0
        assert encoded[0, 0, 1] == 255


class TestCameraPath:
    def test_requires_increasing_times(self):
        with pytest.raises(ValueError):
            CameraPath(
                [
                    CameraKeyframe(0.0, 20.0, 80.0),
                    CameraKeyframe(0.0, 25.0, 80.0),
                ]
            )

    def test_endpoint_poses(self):
        path = orbit_path(10.0, 25.0, 80.0, revolutions=2.0)
        assert path.pose_at(0.0).azimuth_degrees == 0.0
        assert path.pose_at(10.0).azimuth_degrees == 720.0
        assert path.pose_at(-1.0).distance == 25.0
        assert path.pose_at(11.0).azimuth_degrees == 720.0

    def test_midpoint_blend(self):
        path = CameraPath(
            [
                CameraKeyframe(0.0, 20.0, 90.0, 0.0),
                CameraKeyframe(4.0, 40.0, 70.0, 180.0),
            ]
        )
        pose = path.pose_at(2.0)
        assert abs(pose.distance - 30.0) < 1e-9
        assert abs(pose.azimuth_degrees - 90.0) < 1e-9

    def test_camera_orbit_radius_constant(self):
        path = orbit_path(6.0, 30.0, 85.0)
        for t in numpy.linspace(0.0, 6.0, 7):
            camera = path.camera_at(float(t))
            assert abs(camera.distance_from_origin - 30.0) < 1e-9


class TestOfflineRender:
    def test_subpixel_directions_shape_and_norm(self):
        camera = FlyCamera.from_orbit(25.0, 80.0)
        directions = subpixel_directions(camera, 8, 6, 70.0, 2)
        assert directions.shape == (8 * 6 * 4, 3)
        norms = numpy.linalg.norm(directions, axis=-1)
        numpy.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_small_frame_physics(self):
        """A small offline frame shows shadow, disk, and stars."""
        settings = OfflineSettings(
            spin=0.9, supersample=1, max_steps=5000, tile_rays=4000
        )
        camera = FlyCamera.from_orbit(26.0, 82.0)
        hdr = render_hdr(camera, 96, 60, settings, progress=False)
        assert hdr.shape == (60, 96, 3)
        assert bool(numpy.isfinite(hdr).all())
        image = develop(hdr, exposure=1.4)
        black = numpy.all(image < 10, axis=-1)
        warm = (
            image[:, :, 0].astype(int) - image[:, :, 2].astype(int)
        ) > 15
        assert 0.005 < black.mean() < 0.6, "shadow must exist"
        assert warm.mean() > 0.1, "disk must dominate this view"
        assert float(hdr.max()) > 1.0, "inner disk must exceed unit HDR"

    def test_render_is_deterministic(self):
        settings = OfflineSettings(
            spin=0.5, supersample=1, max_steps=3000, tile_rays=3000
        )
        camera = FlyCamera.from_orbit(28.0, 78.0)
        first = render_hdr(camera, 48, 30, settings, progress=False)
        second = render_hdr(camera, 48, 30, settings, progress=False)
        numpy.testing.assert_array_equal(first, second)

    def test_supersampling_reduces_edge_noise(self):
        """More subpixel samples smooth the disk rim."""
        camera = FlyCamera.from_orbit(26.0, 82.0)
        base = OfflineSettings(
            spin=0.9, supersample=1, max_steps=4000, tile_rays=8000
        )
        smooth = OfflineSettings(
            spin=0.9, supersample=2, max_steps=4000, tile_rays=8000
        )
        rough_hdr = render_hdr(camera, 48, 30, base, progress=False)
        smooth_hdr = render_hdr(camera, 48, 30, smooth, progress=False)

        def gradient_energy(hdr):
            gx = numpy.diff(hdr, axis=1)
            gy = numpy.diff(hdr, axis=0)
            return float(numpy.abs(gx).mean() + numpy.abs(gy).mean())

        assert gradient_energy(smooth_hdr) < gradient_energy(rough_hdr)


class TestVideo:
    def test_downsample_box_average(self):
        image = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
        image[:2, :2] = 100
        small = downsample(image, 2)
        assert small.shape == (2, 2, 3)
        assert int(small[0, 0, 0]) == 100
        assert int(small[1, 1, 0]) == 0

    def test_frame_bloom_zero_strength_is_identity(self):
        rng = numpy.random.default_rng(1)
        image = rng.integers(0, 255, (24, 24, 3), dtype=numpy.uint8)
        numpy.testing.assert_array_equal(frame_bloom(image, 0.0), image)

    def test_encode_frames(self, tmp_path):
        pytest.importorskip("imageio")
        pytest.importorskip("imageio_ffmpeg")
        from PIL import Image

        from blackhorizon.offline.video import encode_frames

        for index in range(4):
            frame = numpy.full(
                (32, 48, 3), index * 40, dtype=numpy.uint8
            )
            Image.fromarray(frame).save(
                tmp_path / f"frame_{index:03d}.png"
            )
        output = tmp_path / "clip.mp4"
        count = encode_frames(str(tmp_path), str(output), fps=8)
        assert count == 4
        assert output.exists()
        assert output.stat().st_size > 500

    def test_encode_frames_empty_dir(self, tmp_path):
        pytest.importorskip("imageio")
        from blackhorizon.offline.video import encode_frames

        with pytest.raises(FileNotFoundError):
            encode_frames(str(tmp_path), str(tmp_path / "x.mp4"), fps=8)
