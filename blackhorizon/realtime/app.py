"""Interactive real-time black hole viewer.

Usage:
    python -m blackhorizon.realtime.app [--spin 0.9] [--width 1280] ...

Controls:
    W / A / S / D        move forward / left / back / right
    Q / E                move down / up
    Left shift           speed boost
    Right mouse drag     look around
    R                    reset camera to the starting orbit
    1 / 2 / 3 / 4        quality presets low / medium / high / ultra
    B                    toggle background style
    F12                  save a screenshot (screenshot_NNN.png)
    Escape               quit

If imgui-bundle is installed, a settings panel exposes spin, field of
view, quality, and resolution scale (rendered via imgui-bundle's native
backends, so no further dependencies are needed); without it the app
still runs with the keyboard controls above plus [ and ] to adjust spin.
"""

from __future__ import annotations

import argparse
import time

from .glfw_compat import ensure_compatible_glfw

# Must run before the first import of glfw so that pyGLFW and
# imgui-bundle share one GLFW library; see glfw_compat for the details.
ensure_compatible_glfw()

import glfw
import moderngl
import numpy

from ..imaging import save_png
from .engine import KerrRenderEngine
from .fly_camera import FlyCamera
from .settings import BackgroundMode, QualityPreset, RenderSettings


_BLIT_VERTEX = """
#version 330 core
out vec2 v_uv;
void main() {
    vec2 corners[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    v_uv = corners[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(corners[gl_VertexID], 0.0, 1.0);
}
"""

_BLIT_FRAGMENT = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
void main() {
    frag_color = texture(u_texture, v_uv);
}
"""


class InteractiveApp:
    """Owns the window, the render loop, input handling, and the UI."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not glfw.init():
            raise RuntimeError("glfw initialization failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        self.window = glfw.create_window(
            args.width, args.height, "Black Horizon", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError(
                "window creation failed; on Wayland try GLFW_PLATFORM=x11"
            )
        glfw.make_context_current(self.window)
        glfw.swap_interval(0 if args.no_vsync else 1)

        self.ctx = moderngl.create_context()
        self.engine = KerrRenderEngine(self.ctx)
        self.blit_program = self.ctx.program(
            vertex_shader=_BLIT_VERTEX, fragment_shader=_BLIT_FRAGMENT
        )
        self.blit_vao = self.ctx.vertex_array(self.blit_program, [])
        self.blit_vao.vertices = 3

        self.settings = RenderSettings(spin=args.spin).apply_preset(
            QualityPreset(args.quality)
        )
        self.start_distance = args.distance
        self.start_inclination = args.inclination
        self.camera = FlyCamera.from_orbit(args.distance, args.inclination)

        self.panel = None
        if not args.no_ui:
            try:
                from .ui import SettingsPanel

                self.panel = SettingsPanel(self.window)
            except Exception as exc:
                print(
                    f"settings panel disabled ({exc}); running with "
                    "keyboard controls only"
                )

        self._last_cursor = None
        self._screenshot_index = 0
        self._pressed_once: set[int] = set()
        self._fps = 0.0

    # Input handling

    def _key_pressed_once(self, key: int) -> bool:
        """True exactly once per physical key press (edge detection)."""
        if glfw.get_key(self.window, key) == glfw.PRESS:
            if key not in self._pressed_once:
                self._pressed_once.add(key)
                return True
            return False
        self._pressed_once.discard(key)
        return False

    def _ui_wants_input(self) -> tuple[bool, bool]:
        """Whether the settings panel currently captures (keyboard, mouse)."""
        if self.panel is None:
            return False, False
        return self.panel.want_capture()

    def _handle_input(self, dt: float) -> None:
        ui_keyboard, ui_mouse = self._ui_wants_input()

        if not ui_keyboard:
            direction = numpy.zeros(3)
            key = lambda k: glfw.get_key(self.window, k) == glfw.PRESS
            if key(glfw.KEY_W):
                direction[0] += 1.0
            if key(glfw.KEY_S):
                direction[0] -= 1.0
            if key(glfw.KEY_D):
                direction[1] += 1.0
            if key(glfw.KEY_A):
                direction[1] -= 1.0
            if key(glfw.KEY_E):
                direction[2] += 1.0
            if key(glfw.KEY_Q):
                direction[2] -= 1.0
            boost = 4.0 if key(glfw.KEY_LEFT_SHIFT) else 1.0
            self.camera.move(direction, dt, boost)

            if self._key_pressed_once(glfw.KEY_R):
                self.camera = FlyCamera.from_orbit(
                    self.start_distance, self.start_inclination
                )
            if self._key_pressed_once(glfw.KEY_B):
                self.settings.background = (
                    BackgroundMode.STARFIELD
                    if self.settings.background == BackgroundMode.CHECKERBOARD
                    else BackgroundMode.CHECKERBOARD
                )
            presets = {
                glfw.KEY_1: QualityPreset.LOW,
                glfw.KEY_2: QualityPreset.MEDIUM,
                glfw.KEY_3: QualityPreset.HIGH,
                glfw.KEY_4: QualityPreset.ULTRA,
            }
            for key_code, preset in presets.items():
                if self._key_pressed_once(key_code):
                    self.settings = self.settings.apply_preset(preset)
            if self._key_pressed_once(glfw.KEY_LEFT_BRACKET):
                self.settings.spin = max(-1.0, self.settings.spin - 0.05)
            if self._key_pressed_once(glfw.KEY_RIGHT_BRACKET):
                self.settings.spin = min(1.0, self.settings.spin + 0.05)
            if self._key_pressed_once(glfw.KEY_F12):
                self._save_screenshot()

        if not ui_mouse and glfw.get_mouse_button(
            self.window, glfw.MOUSE_BUTTON_RIGHT
        ) == glfw.PRESS:
            cursor = glfw.get_cursor_pos(self.window)
            if self._last_cursor is not None:
                dx = cursor[0] - self._last_cursor[0]
                dy = cursor[1] - self._last_cursor[1]
                self.camera.rotate(dx, dy)
            self._last_cursor = cursor
        else:
            self._last_cursor = None

    def _save_screenshot(self) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        render_w, render_h = self.engine.render_size(
            width, height, self.settings
        )
        image = self.engine.read_frame(
            self.settings, self.camera, render_w, render_h
        )
        path = f"screenshot_{self._screenshot_index:03d}.png"
        save_png(image, path)
        self._screenshot_index += 1
        print(f"Saved {path}")

    # UI panel

    def _draw_ui(self) -> None:
        if self.panel is None:
            return
        self.settings = self.panel.draw(
            self.settings, self.camera, self._fps
        )

    # Main loop

    def run(self) -> None:
        """Run the render loop until the window closes."""
        previous = time.perf_counter()
        smoothed_dt = 1.0 / 60.0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt = min(now - previous, 0.1)
            previous = now
            smoothed_dt = 0.95 * smoothed_dt + 0.05 * dt
            self._fps = 1.0 / max(smoothed_dt, 1e-6)

            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)
            self._handle_input(dt)

            width, height = glfw.get_framebuffer_size(self.window)
            if width == 0 or height == 0:
                continue
            render_w, render_h = self.engine.render_size(
                width, height, self.settings
            )
            texture = self.engine.render(
                self.settings, self.camera, render_w, render_h
            )

            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, width, height)
            texture.use(location=0)
            self.blit_program["u_texture"].value = 0
            self.blit_vao.render(moderngl.TRIANGLES)

            self._draw_ui()
            glfw.swap_buffers(self.window)

        self._shutdown()

    def _shutdown(self) -> None:
        if self.panel is not None:
            self.panel.shutdown()
        self.engine.release()
        glfw.terminate()


def parse_args() -> argparse.Namespace:
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        description="Interactive real-time Kerr black hole viewer."
    )
    parser.add_argument("--spin", type=float, default=0.9)
    parser.add_argument("--distance", type=float, default=30.0)
    parser.add_argument("--inclination", type=float, default=85.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--quality",
        choices=[preset.value for preset in QualityPreset],
        default="medium",
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="disable the imgui panel"
    )
    parser.add_argument(
        "--no-vsync", action="store_true",
        help="disable vsync to measure uncapped frame rates",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point of the interactive viewer."""
    app = InteractiveApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
