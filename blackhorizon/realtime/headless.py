"""Render one frame of the real-time tracer without opening a window.

Usage:
    python -m blackhorizon.realtime.headless --spin 0.9 --output frame.png

Creates a standalone OpenGL context, renders a single frame with the same
shader the interactive app uses, and writes it to a PNG. This is the
quickest way to verify the GL path on a new machine and is what the GL
test suite uses.
"""

from __future__ import annotations

import argparse
import time

import moderngl

from ..imaging import save_png
from .engine import KerrRenderEngine
from .fly_camera import FlyCamera
from .settings import BackgroundMode, QualityPreset, RenderSettings


def create_standalone_context() -> moderngl.Context:
    """Create a windowless OpenGL context, trying EGL before the default."""
    try:
        return moderngl.create_context(standalone=True, backend="egl")
    except Exception:
        return moderngl.create_context(standalone=True)


def render_frame(
    settings: RenderSettings,
    camera: FlyCamera,
    width: int,
    height: int,
    ctx: moderngl.Context | None = None,
):
    """Render a single frame headlessly and return it as an image array."""
    owns_context = ctx is None
    if owns_context:
        ctx = create_standalone_context()
    engine = KerrRenderEngine(ctx)
    try:
        return engine.read_frame(settings, camera, width, height)
    finally:
        engine.release()
        if owns_context:
            ctx.release()


def main() -> None:
    """Parse arguments, render one frame, and save it."""
    parser = argparse.ArgumentParser(
        description="Headless single-frame render of the real-time tracer."
    )
    parser.add_argument("--spin", type=float, default=0.9)
    parser.add_argument("--inclination", type=float, default=85.0)
    parser.add_argument("--distance", type=float, default=30.0)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--quality",
        choices=[preset.value for preset in QualityPreset],
        default="high",
    )
    parser.add_argument(
        "--background", choices=("checkerboard", "starfield"),
        default="starfield",
    )
    parser.add_argument("--no-disk", action="store_true")
    parser.add_argument("--disk-outer", type=float, default=18.0)
    parser.add_argument("--disk-temperature", type=float, default=6500.0)
    parser.add_argument("--exposure", type=float, default=1.5)
    parser.add_argument("--output", type=str, default="frame.png")
    args = parser.parse_args()

    settings = RenderSettings(
        spin=args.spin,
        fov_degrees=args.fov,
        resolution_scale=1.0,
        background=BackgroundMode.CHECKERBOARD
        if args.background == "checkerboard"
        else BackgroundMode.STARFIELD,
        disk_enabled=not args.no_disk,
        disk_outer_radius=args.disk_outer,
        disk_temperature=args.disk_temperature,
        exposure=args.exposure,
    ).apply_preset(QualityPreset(args.quality))
    settings.resolution_scale = 1.0
    camera = FlyCamera.from_orbit(args.distance, args.inclination)

    start = time.perf_counter()
    image = render_frame(settings, camera, args.width, args.height)
    elapsed = time.perf_counter() - start
    save_png(image, args.output)
    print(
        f"Rendered {args.width}x{args.height} spin={args.spin} "
        f"quality={args.quality} in {elapsed:.2f} s, wrote {args.output}"
    )


if __name__ == "__main__":
    main()
