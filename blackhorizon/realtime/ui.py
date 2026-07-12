"""Settings panel built on imgui-bundle's native backends.

Uses the GLFW and OpenGL3 backends compiled into imgui-bundle itself, so
the only dependency is the imgui-bundle wheel; in particular PyOpenGL is
not required (the pure-Python example backends shipped with imgui-bundle
do require it, which is why they are not used here).

The panel owns the imgui context and backend lifecycles; the app calls
want_capture before handling its own input, draw once per frame after
the scene, and shutdown on exit.
"""

from __future__ import annotations

import ctypes

from imgui_bundle import imgui

from .fly_camera import FlyCamera
from .settings import QualityPreset, RenderSettings


class SettingsPanel:
    """Immediate-mode settings window rendered over the scene."""

    def __init__(self, glfw_window) -> None:
        """Attach imgui to an existing GLFW window with a GL 3.3 context.

        Args:
            glfw_window: The window handle returned by glfw.create_window.

        Raises:
            RuntimeError: If the imgui backends fail to initialize.
        """
        imgui.create_context()
        imgui.get_io().set_ini_filename("")
        window_address = ctypes.cast(glfw_window, ctypes.c_void_p).value
        if not imgui.backends.glfw_init_for_opengl(window_address, True):
            raise RuntimeError("imgui glfw backend initialization failed")
        if not imgui.backends.opengl3_init("#version 330"):
            imgui.backends.glfw_shutdown()
            raise RuntimeError("imgui opengl3 backend initialization failed")

    def want_capture(self) -> tuple[bool, bool]:
        """Whether imgui currently captures (keyboard, mouse) input."""
        io = imgui.get_io()
        return io.want_capture_keyboard, io.want_capture_mouse

    def draw(
        self, settings: RenderSettings, camera: FlyCamera, fps: float
    ) -> RenderSettings:
        """Draw the panel and return the (possibly preset-replaced) settings.

        Scalar fields are mutated in place; applying a quality preset
        returns a new RenderSettings instance, which the caller must
        adopt.
        """
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        imgui.begin("Black Horizon")
        imgui.text(f"{fps:.0f} fps")
        imgui.text(f"camera r = {camera.distance_from_origin:.1f} M")

        changed, value = imgui.slider_float(
            "spin a/M", settings.spin, -0.999, 0.999
        )
        if changed:
            settings.spin = value
        changed, value = imgui.slider_float(
            "fov deg", settings.fov_degrees, 20.0, 120.0
        )
        if changed:
            settings.fov_degrees = value
        changed, value = imgui.slider_int(
            "max steps", settings.max_steps, 64, 4096
        )
        if changed:
            settings.max_steps = value
        changed, value = imgui.slider_float(
            "step scale", settings.step_scale, 0.02, 0.5
        )
        if changed:
            settings.step_scale = value
        changed, value = imgui.slider_float(
            "resolution scale", settings.resolution_scale, 0.25, 1.0
        )
        if changed:
            settings.resolution_scale = value

        for preset in QualityPreset:
            if imgui.button(preset.value):
                settings = settings.apply_preset(preset)
            imgui.same_line()
        imgui.new_line()
        imgui.text("right-drag: look, WASDQE: move, shift: boost")
        imgui.end()

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        return settings

    def shutdown(self) -> None:
        """Release the imgui backends and context."""
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
