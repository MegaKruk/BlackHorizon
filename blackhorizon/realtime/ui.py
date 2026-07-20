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

import math

import numpy

from ..emission.bluesheet import blueshift_amplification
from ..kerr import KerrSpacetime
from .fly_camera import FlyCamera
from .settings import BackgroundMode, QualityPreset, RenderSettings


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
        self,
        settings: RenderSettings,
        camera: FlyCamera,
        fps: float,
        infall=None,
        overlay_enabled: bool = True,
        time_scale: float = 0.2,
    ) -> tuple[RenderSettings, dict]:
        """Draw the panel; return settings and requested actions.

        Scalar fields are mutated in place; applying a quality preset
        returns a new RenderSettings instance, which the caller must
        adopt. The actions dict carries the reset request and the
        overlay and time-scale values.
        """
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        actions = {
            "reset": False,
            "overlay_enabled": overlay_enabled,
            "time_scale": time_scale,
        }

        imgui.begin("Black Horizon")
        imgui.text(f"{fps:.0f} fps")
        if infall is None:
            imgui.text(
                f"camera r = {camera.distance_from_origin:.1f} M"
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.35, 0.25, 1.0),
                f"INSIDE THE HORIZON  r = {infall.radius:.3f} M",
            )
            inner = 1.0 - math.sqrt(
                max(0.0, 1.0 - settings.spin * settings.spin)
            )
            if (
                settings.interior_journey == "realistic"
                and inner > 1e-9
                and not infall.terminated()
            ):
                amplification = float(
                    blueshift_amplification(
                        KerrSpacetime(spin=settings.spin),
                        numpy.array([infall.radius]),
                    )[0]
                )
                if amplification > 1.001:
                    imgui.text_colored(
                        imgui.ImVec4(0.5, 0.85, 1.0, 1.0),
                        "blue sheet amplification: "
                        f"{amplification:6.1f} x",
                    )
            if (
                settings.interior_journey == "idealized"
                and inner > 0.0
                and infall.radius < inner
            ):
                imgui.text_colored(
                    imgui.ImVec4(0.4, 0.85, 1.0, 1.0),
                    "beyond the Cauchy horizon (idealized vacuum "
                    "Kerr)",
                )
            if infall.terminated():
                if infall.termination_reason == "chart":
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.2, 0.2, 1.0),
                        "chart boundary: ring plane or Cauchy "
                        "horizon exit",
                    )
                elif (
                    settings.interior_journey == "realistic"
                    and inner > 1e-9
                ):
                    imgui.text_colored(
                        imgui.ImVec4(0.7, 0.9, 1.0, 1.0),
                        "the blue sheet: infalling radiation "
                        "blueshifted beyond survival",
                    )
                else:
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.2, 0.2, 1.0),
                        "worldline ended at the terminal surface",
                    )
            else:
                if infall.remaining_tau >= 59.9:
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.55, 0.2, 1.0),
                        "no terminal surface on current path",
                    )
                else:
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.55, 0.2, 1.0),
                        "proper time remaining: "
                        f"{infall.remaining_tau:6.3f} M",
                    )
                imgui.text(
                    f"proper time inside: {infall.elapsed_tau:6.3f} M"
                )
            changed, value = imgui.slider_float(
                "time scale M/s", time_scale, 0.02, 1.0
            )
            if changed:
                actions["time_scale"] = value
        if imgui.button("reset camera"):
            actions["reset"] = True
        imgui.same_line()
        starfield = settings.background == BackgroundMode.STARFIELD
        changed, value = imgui.checkbox("starfield", starfield)
        if changed:
            settings.background = (
                BackgroundMode.STARFIELD
                if value
                else BackgroundMode.CHECKERBOARD
            )
        imgui.same_line()
        changed, value = imgui.checkbox("overlays", overlay_enabled)
        if changed:
            actions["overlay_enabled"] = value
        imgui.text("journey past the Cauchy horizon:")
        if imgui.radio_button(
            "realistic (blue sheet)",
            settings.interior_journey == "realistic",
        ):
            settings.interior_journey = "realistic"
        imgui.same_line()
        if imgui.radio_button(
            "idealized Kerr",
            settings.interior_journey == "idealized",
        ):
            settings.interior_journey = "idealized"
        if overlay_enabled:
            self._draw_light_cone_glyph(settings, camera, infall)

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

        imgui.separator()
        changed, value = imgui.checkbox("accretion disk", settings.disk_enabled)
        if changed:
            settings.disk_enabled = value
        changed, value = imgui.slider_float(
            "disk outer r/M", settings.disk_outer_radius, 3.0, 40.0
        )
        if changed:
            settings.disk_outer_radius = value
        changed, value = imgui.slider_float(
            "disk peak T (K)", settings.disk_temperature, 1000.0, 20000.0
        )
        if changed:
            settings.disk_temperature = value
        changed, value = imgui.slider_float(
            "exposure", settings.exposure, 0.1, 5.0
        )
        if changed:
            settings.exposure = value
        changed, value = imgui.slider_float(
            "disk detail", settings.disk_detail, 0.0, 1.0
        )
        if changed:
            settings.disk_detail = value

        for preset in QualityPreset:
            if imgui.button(preset.value):
                settings = settings.apply_preset(preset)
            imgui.same_line()
        imgui.new_line()
        if infall is None:
            imgui.text("right-drag: look, WASDQE: move, shift: boost")
        else:
            imgui.text(
                "right-drag: look, WASDQE: thrust (burns shorten "
                "life), R: reset"
            )
        imgui.end()

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        return settings, actions

    def _draw_light_cone_glyph(
        self, settings: RenderSettings, camera: FlyCamera, infall
    ) -> None:
        """Pedagogical light-cone widget in the panel.

        Draws the future light cone tilted by the river-model inflow
        speed v = sqrt(2H) (Hamilton and Lisle 2008): v < 1 outside,
        v = 1 at the horizon, v > 1 inside, where the whole cone points
        inward and no escape direction exists.
        """
        radius = (
            infall.radius if infall is not None
            else camera.distance_from_origin
        )
        spin = settings.spin
        r_plus = 1.0 + math.sqrt(max(0.0, 1.0 - spin * spin))
        # Equatorial Kerr-Schild H = M r / (r^2 + small): use the
        # Schwarzschild-form magnitude as the glyph's inflow speed.
        v_river = math.sqrt(2.0 / max(radius, 1e-3))
        imgui.text(
            f"river inflow v = {v_river:.2f} c "
            + ("(inside: no way out)" if radius <= r_plus else "")
        )
        draw_list = imgui.get_window_draw_list()
        origin = imgui.get_cursor_screen_pos()
        size = 84.0
        apex = imgui.ImVec2(origin.x + size, origin.y + 8.0)
        length = size * 0.75
        # The cone axis tilts inward (screen left) by the inflow speed;
        # edges sit 45 degrees either side of the axis.
        tilt = math.atan(v_river)
        for sign in (-1.0, 1.0):
            angle = tilt + sign * (math.pi / 4.0)
            tip = imgui.ImVec2(
                apex.x - length * math.sin(angle),
                apex.y + length * math.cos(angle),
            )
            draw_list.add_line(
                apex, tip, imgui.color_convert_float4_to_u32(
                    imgui.ImVec4(1.0, 0.8, 0.3, 1.0)
                ), 2.0,
            )
        label = (
            "both cone edges point inward"
            if v_river > 1.0
            else "outward edge still escapes"
        )
        imgui.dummy(imgui.ImVec2(size * 2.2, length + 14.0))
        imgui.text(label)

    def shutdown(self) -> None:
        """Release the imgui backends and context."""
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
