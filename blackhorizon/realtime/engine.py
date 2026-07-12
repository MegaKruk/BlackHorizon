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

    def release(self) -> None:
        """Free all GPU resources owned by the engine."""
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._color_texture.release()
            self._framebuffer = None
            self._color_texture = None
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

    def _set_uniforms(
        self, settings: RenderSettings, camera: FlyCamera, aspect: float
    ) -> None:
        """Push all shader uniforms for the current frame."""
        spacetime = KerrSpacetime(mass=1.0, spin=settings.spin)
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
