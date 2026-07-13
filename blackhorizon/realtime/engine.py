"""ModernGL render engine for the real-time Kerr tracer.

Owns the GPU resources: the compiled shader program, the fullscreen
vertex array, and an offscreen framebuffer at the internal render
resolution. The windowed app blits the offscreen texture to the screen;
headless callers read it back as an array. All physics parameters arrive
per frame via RenderSettings and FlyCamera, keeping this class free of
simulation logic.
"""

from __future__ import annotations

import math

import moderngl
import numpy

from ..emission.blackbody import blackbody_lut
from ..emission.novikov_thorne import disk_inner_radius, temperature_lut
from ..kerr import KerrSpacetime
from .fly_camera import FlyCamera
from .settings import MAX_STEPS_HARD_LIMIT, RenderSettings
from .shader_source import fragment_source, vertex_source


class KerrRenderEngine:
    """Renders the lensed Kerr view with a fullscreen fragment shader."""

    def __init__(self, ctx: moderngl.Context) -> None:
        """Compile the tracer program and create the fullscreen geometry.

        Args:
            ctx: An existing ModernGL context (windowed or standalone).
        """
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=vertex_source(),
            fragment_shader=fragment_source(MAX_STEPS_HARD_LIMIT),
        )
        self.vao = ctx.vertex_array(self.program, [])
        self.vao.vertices = 3
        self._color_texture: moderngl.Texture | None = None
        self._framebuffer: moderngl.Framebuffer | None = None
        self._size: tuple[int, int] = (0, 0)
        self._temperature_texture: moderngl.Texture | None = None
        self._temperature_key: tuple[float, float] | None = None
        self._disk_inner: float = 0.0
        bb_table, self._bb_log_min, self._bb_log_max = blackbody_lut()
        self._blackbody_texture = ctx.texture(
            (bb_table.shape[0], 1), 3, bb_table.tobytes(), dtype="f4"
        )
        self._blackbody_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def release(self) -> None:
        """Free all GPU resources owned by the engine."""
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._color_texture.release()
            self._framebuffer = None
            self._color_texture = None
        if self._temperature_texture is not None:
            self._temperature_texture.release()
            self._temperature_texture = None
        self._blackbody_texture.release()
        self.vao.release()
        self.program.release()

    @property
    def texture(self) -> moderngl.Texture | None:
        """Offscreen color texture of the last render, if any."""
        return self._color_texture

    def _ensure_target(self, width: int, height: int) -> None:
        """(Re)create the offscreen target when the size changes."""
        if (width, height) == self._size and self._framebuffer is not None:
            return
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._color_texture.release()
        self._color_texture = self.ctx.texture((width, height), 3)
        self._color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._framebuffer = self.ctx.framebuffer(
            color_attachments=[self._color_texture]
        )
        self._size = (width, height)

    def _effective_disk_radii(
        self, spacetime: KerrSpacetime, settings: RenderSettings
    ) -> tuple[float, float]:
        """Inner (ISCO) and clamped outer disk radii for the frame."""
        inner = disk_inner_radius(spacetime)
        outer = max(settings.disk_outer_radius, inner + 1.0)
        return inner, outer

    def _ensure_temperature_lut(
        self, spacetime: KerrSpacetime, settings: RenderSettings
    ) -> None:
        """(Re)build the T(r) lookup texture when spin or size change."""
        inner, outer = self._effective_disk_radii(spacetime, settings)
        key = (round(spacetime.spin, 4), round(outer, 3))
        if key == self._temperature_key:
            return
        table, r_inner, _ = temperature_lut(spacetime.spin, outer)
        if self._temperature_texture is not None:
            self._temperature_texture.release()
        self._temperature_texture = self.ctx.texture(
            (table.shape[0], 1), 1, table.tobytes(), dtype="f4"
        )
        self._temperature_texture.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR,
        )
        self._temperature_key = key
        self._disk_inner = r_inner

    def _observer_lapse(
        self, spacetime: KerrSpacetime, camera: FlyCamera
    ) -> float:
        """Static-observer lapse at the camera, clamped inside the
        ergosphere where no static frame exists."""
        geo = spacetime.geometry(
            numpy.asarray([camera.position[0]]),
            numpy.asarray([camera.position[1]]),
            numpy.asarray([camera.position[2]]),
        )
        g_tt = 2.0 * float(geo.h[0]) - 1.0
        if g_tt >= -1e-3:
            return 1.0
        return 1.0 / math.sqrt(-g_tt)

    def _set_uniforms(
        self, settings: RenderSettings, camera: FlyCamera, aspect: float
    ) -> None:
        """Push all shader uniforms for the current frame."""
        spacetime = KerrSpacetime(mass=1.0, spin=settings.spin)
        self._ensure_temperature_lut(spacetime, settings)
        inner, outer = self._effective_disk_radii(spacetime, settings)
        self._temperature_texture.use(location=0)
        self._blackbody_texture.use(location=1)
        forward, right, up = camera.basis()
        uniforms = {
            "u_cam_position": tuple(camera.position.astype(numpy.float32)),
            "u_cam_forward": tuple(forward.astype(numpy.float32)),
            "u_cam_right": tuple(right.astype(numpy.float32)),
            "u_cam_up": tuple(up.astype(numpy.float32)),
            "u_tan_half_fov": math.tan(
                math.radians(settings.fov_degrees) / 2.0
            ),
            "u_aspect": aspect,
            "u_spin": settings.spin,
            "u_horizon_radius": spacetime.outer_horizon_radius,
            "u_escape_radius": settings.effective_escape_radius(
                camera.distance_from_origin
            ),
            "u_max_steps": settings.max_steps,
            "u_step_scale": settings.step_scale,
            "u_min_step": settings.min_step,
            "u_max_step": settings.max_step,
            "u_capture_margin": settings.capture_margin,
            "u_momentum_bailout": settings.momentum_bailout,
            "u_background_mode": int(settings.background),
            "u_disk_enabled": 1 if settings.disk_enabled else 0,
            "u_disk_inner_radius": inner,
            "u_disk_outer_radius": outer,
            "u_disk_peak_temperature": settings.disk_temperature,
            "u_disk_detail": settings.disk_detail,
            "u_exposure": settings.exposure,
            "u_observer_lapse": self._observer_lapse(spacetime, camera),
            "u_bb_log_min": self._bb_log_min,
            "u_bb_log_max": self._bb_log_max,
            "u_temperature_lut": 0,
            "u_blackbody_lut": 1,
        }
        for name, value in uniforms.items():
            self.program[name].value = value

    def render_size(
        self, window_width: int, window_height: int, settings: RenderSettings
    ) -> tuple[int, int]:
        """Internal render resolution for a window size and settings."""
        scale = settings.resolution_scale
        return (
            max(1, int(window_width * scale)),
            max(1, int(window_height * scale)),
        )

    def render(
        self,
        settings: RenderSettings,
        camera: FlyCamera,
        width: int,
        height: int,
    ) -> moderngl.Texture:
        """Render one frame to the offscreen texture and return it."""
        settings.validate()
        self._ensure_target(width, height)
        self._set_uniforms(settings, camera, width / height)
        self._framebuffer.use()
        self.ctx.viewport = (0, 0, width, height)
        self.vao.render(moderngl.TRIANGLES)
        return self._color_texture

    def read_frame(
        self,
        settings: RenderSettings,
        camera: FlyCamera,
        width: int,
        height: int,
    ) -> numpy.ndarray:
        """Render one frame and read it back as a (height, width, 3) array.

        The row order is flipped from OpenGL's bottom-up convention to the
        image convention with row 0 at the top.
        """
        self.render(settings, camera, width, height)
        raw = self._framebuffer.read(components=3)
        image = numpy.frombuffer(raw, dtype=numpy.uint8)
        return image.reshape(height, width, 3)[::-1].copy()
