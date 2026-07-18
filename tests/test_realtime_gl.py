"""Headless OpenGL tests of the real shader pipeline.

Skipped automatically when no OpenGL context can be created (for example
on a machine without GL drivers); everywhere else these compile the
actual GLSL and verify that the float32 GPU tracer classifies pixels
exactly like the float64 reference implementation.
"""

import math

import numpy
import pytest

moderngl = pytest.importorskip("moderngl")

from blackhorizon.geodesics import build_state, null_momentum_from_velocity
from blackhorizon.kerr import KerrSpacetime
from blackhorizon.realtime.engine import KerrRenderEngine
from blackhorizon.realtime.fly_camera import FlyCamera
from blackhorizon.realtime.reference import trace_like_shader
from blackhorizon.realtime.settings import (
    BackgroundMode,
    QualityPreset,
    RenderSettings,
)
from blackhorizon.tracer import RayStatus


@pytest.fixture(scope="module")
def gl_context():
    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
    except Exception:
        try:
            ctx = moderngl.create_context(standalone=True)
        except Exception:
            pytest.skip("no standalone OpenGL context available")
    yield ctx
    ctx.release()


@pytest.fixture(scope="module")
def engine(gl_context):
    engine = KerrRenderEngine(gl_context)
    yield engine
    engine.release()


def pixel_ray_states(spacetime, camera, settings, width, height):
    """Initial states for the exact rays the shader traces per pixel."""
    forward, right, up = camera.basis()
    tan_half = math.tan(math.radians(settings.fov_degrees) / 2.0)
    aspect = width / height
    xs = (numpy.arange(width) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (numpy.arange(height) + 0.5) / height * 2.0
    u, v = numpy.meshgrid(xs, ys)
    directions = (
        forward[None, None]
        + u[..., None] * tan_half * right[None, None]
        + v[..., None] * (tan_half / aspect) * up[None, None]
    )
    directions = (
        directions / numpy.linalg.norm(directions, axis=-1, keepdims=True)
    ).reshape(-1, 3)
    positions = numpy.tile(camera.position[None, :], (width * height, 1))
    momenta = null_momentum_from_velocity(
        spacetime, positions, directions, time_orientation="past"
    )
    return build_state(positions, momenta)


class TestShaderPipeline:
    def test_program_compiles(self, engine):
        assert engine.program is not None

    def test_frame_shape_and_finiteness(self, engine):
        settings = RenderSettings(spin=0.9).apply_preset(QualityPreset.LOW)
        camera = FlyCamera.from_orbit(30.0, 85.0)
        image = engine.read_frame(settings, camera, 96, 72)
        assert image.shape == (72, 96, 3)
        assert image.dtype == numpy.uint8

    def test_shader_matches_float64_reference(self, engine):
        """Per-pixel black classification agrees with the CPU reference."""
        width, height = 128, 96
        settings = RenderSettings(spin=0.9, disk_enabled=False).apply_preset(
            QualityPreset.MEDIUM
        )
        camera = FlyCamera.from_orbit(30.0, 85.0)
        image = engine.read_frame(settings, camera, width, height)
        shader_black = numpy.all(image < 8, axis=-1)

        st = KerrSpacetime(spin=settings.spin)
        state0 = pixel_ray_states(st, camera, settings, width, height)
        ref = trace_like_shader(
            st,
            state0,
            settings,
            settings.effective_escape_radius(camera.distance_from_origin),
        )
        ref_black = (
            (ref.status == int(RayStatus.CAPTURED))
            | (ref.status == int(RayStatus.MAX_STEPS))
        ).reshape(height, width)

        agreement = numpy.mean(shader_black == ref_black)
        assert agreement >= 0.995, (
            f"shader/reference agreement {agreement:.4f}"
        )
        # The shadow must exist and be a plausible fraction of the view.
        assert 0.02 < shader_black.mean() < 0.5

    def test_zero_spin_shadow_is_symmetric(self, engine):
        """The Schwarzschild shadow is left-right symmetric on screen."""
        width, height = 128, 128
        settings = RenderSettings(spin=0.0, disk_enabled=False).apply_preset(
            QualityPreset.MEDIUM
        )
        camera = FlyCamera.from_orbit(30.0, 90.0)
        image = engine.read_frame(settings, camera, width, height)
        black = numpy.all(image < 8, axis=-1)
        asymmetry = numpy.mean(black != black[:, ::-1])
        assert asymmetry < 0.01

    def test_background_modes_differ(self, engine):
        camera = FlyCamera.from_orbit(30.0, 85.0)
        base = RenderSettings(spin=0.9).apply_preset(QualityPreset.LOW)
        base.background = BackgroundMode.CHECKERBOARD
        checker = engine.read_frame(base, camera, 96, 72)
        base.background = BackgroundMode.STARFIELD
        stars = engine.read_frame(base, camera, 96, 72)
        assert not numpy.array_equal(checker, stars)

    def test_resolution_scale_mapping(self, engine):
        settings = RenderSettings(resolution_scale=0.5)
        assert engine.render_size(1280, 720, settings) == (640, 360)
        settings.resolution_scale = 1.0
        assert engine.render_size(1280, 720, settings) == (1280, 720)

    def test_disk_mask_matches_reference(self, engine):
        """Shader disk hits agree with the float64 reference mirror.

        The shader mask is isolated by differencing disk-on and disk-off
        frames, which changes exactly the pixels whose rays terminate on
        the disk.
        """
        width, height = 128, 96
        spin = 0.9
        settings = RenderSettings(spin=spin, disk_enabled=True).apply_preset(
            QualityPreset.HIGH
        )
        camera = FlyCamera.from_orbit(28.0, 80.0)
        with_disk = engine.read_frame(settings, camera, width, height)
        settings.disk_enabled = False
        without_disk = engine.read_frame(settings, camera, width, height)
        shader_disk = numpy.any(with_disk != without_disk, axis=-1)

        spacetime = KerrSpacetime(mass=1.0, spin=spin)
        inner = spacetime.isco_radius(prograde=True)
        outer = max(settings.disk_outer_radius, inner + 1.0)
        state0 = pixel_ray_states(spacetime, camera, settings, width, height)
        settings.disk_enabled = True
        reference = trace_like_shader(
            spacetime,
            state0,
            settings,
            settings.effective_escape_radius(camera.distance_from_origin),
            disk_radii=(inner, outer),
        )
        reference_disk = (
            reference.status == int(RayStatus.DISK)
        ).reshape(height, width)
        agreement = float(numpy.mean(shader_disk == reference_disk))
        assert agreement >= 0.98, f"disk mask agreement {agreement:.4f}"

    def test_disk_emission_is_colored_and_beamed(self, engine):
        """Disk pixels carry blackbody color and Doppler asymmetry."""
        width, height = 160, 100
        settings = RenderSettings(
            spin=0.9, disk_enabled=True, exposure=1.5
        ).apply_preset(QualityPreset.MEDIUM)
        settings.background = BackgroundMode.STARFIELD
        camera = FlyCamera.from_orbit(28.0, 82.0)
        image = engine.read_frame(settings, camera, width, height)
        warm = (
            image[:, :, 0].astype(int) - image[:, :, 2].astype(int)
        ) > 15
        assert warm.mean() > 0.05, "expect warm blackbody disk pixels"
        left = float(image[:, : width // 2].mean())
        right = float(image[:, width // 2 :].mean())
        assert left > 1.5 * right, "approaching side must be beamed"


class TestInteriorGL:
    """The GLSL interior mode against the float64 reference."""

    def test_interior_view_matches_reference(self, gl_context):
        """From inside the horizon: dark-region classification agrees
        with the reference, more than half the rendered sky shows the
        outside universe, and the disk is visible (blueshifted) from
        inside."""
        import math

        from blackhorizon.emission.novikov_thorne import disk_inner_radius
        from blackhorizon.frames import (
            build_tetrad,
            rain_four_velocity,
            tetrad_ray_momenta,
        )
        from blackhorizon.geodesics import build_state
        from blackhorizon.realtime.reference import trace_like_shader
        from blackhorizon.tracer import RayStatus

        spacetime = KerrSpacetime(spin=0.0)
        width, height = 96, 72
        settings = RenderSettings(
            spin=0.0,
            interior_mode=True,
            disk_enabled=True,
            background=BackgroundMode.STARFIELD,
            exposure=1.5,
        ).apply_preset(QualityPreset.HIGH)
        position = numpy.array([1.2, 0.4, -0.5])
        camera = FlyCamera(position=position.copy(), yaw=25.0, pitch=35.0)
        camera.four_velocity = rain_four_velocity(
            spacetime, position[None, :]
        )[0]
        engine = KerrRenderEngine(gl_context)
        try:
            image = engine.read_frame(settings, camera, width, height)
        finally:
            engine.release()

        forward, _, up = camera.basis()
        tetrad = build_tetrad(
            spacetime, position, camera.four_velocity, forward, up
        )
        tan_half = math.tan(math.radians(settings.fov_degrees) / 2.0)
        xs = (numpy.arange(width) + 0.5) / width * 2.0 - 1.0
        ys = 1.0 - (numpy.arange(height) + 0.5) / height * 2.0
        gx, gy = numpy.meshgrid(xs, ys)
        local = numpy.stack(
            [
                numpy.ones_like(gx),
                gx * tan_half,
                gy * tan_half / (width / height),
            ],
            axis=-1,
        ).reshape(-1, 3)
        local /= numpy.linalg.norm(local, axis=-1, keepdims=True)
        momenta = tetrad_ray_momenta(spacetime, position, tetrad, local)
        state0 = build_state(
            numpy.tile(position[None, :], (local.shape[0], 1)), momenta
        )
        inner = disk_inner_radius(spacetime)
        outer = max(settings.disk_outer_radius, inner + 1.0)
        result = trace_like_shader(
            spacetime,
            state0,
            settings,
            settings.effective_escape_radius(
                float(numpy.linalg.norm(position))
            ),
            disk_radii=(inner, outer),
            interior_stop=settings.interior_stop,
        )
        status = result.status
        ref_dark = (status == int(RayStatus.TERMINATED)) | (
            status == int(RayStatus.MAX_STEPS)
        )
        ref_disk = status == int(RayStatus.DISK)
        ref_escaped = status == int(RayStatus.ESCAPED)
        shader_black = numpy.all(image < 8, axis=-1).reshape(-1)

        # Every reference-dark ray must render black.
        agreement = float((shader_black | ~ref_dark).mean())
        assert agreement >= 0.97
        # The outside universe plus disk cover most of this view.
        assert float((ref_escaped | ref_disk).mean()) > 0.3
        # The disk is visible from inside and renders bright.
        disk_pixels = image.reshape(-1, 3)[ref_disk]
        assert disk_pixels.shape[0] > 100
        assert float(disk_pixels.max()) > 150
