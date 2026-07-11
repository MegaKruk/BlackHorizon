"""Render the gravitationally lensed shadow of a Kerr black hole.

Usage:
    python -m blackhorizon.examples.render_shadow --spin 0.9 --output shadow.png

Captured photons form the black shadow; escaped photons sample a celestial
checkerboard, making the lensing distortion and the Kerr asymmetry visible.
Use --backend gpu on a machine with CuPy and an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import time

from ..backend import get_xp
from ..camera import PinholeCamera
from ..geodesics import build_state, null_momentum_from_velocity
from ..imaging import render_image, save_png
from ..kerr import KerrSpacetime
from ..tracer import RayStatus, trace_rays


def parse_args() -> argparse.Namespace:
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        description="Render the lensed shadow of a Kerr black hole."
    )
    parser.add_argument(
        "--spin", type=float, default=0.9, help="spin a/M in [-1, 1]"
    )
    parser.add_argument(
        "--inclination",
        type=float,
        default=85.0,
        help="camera inclination from the spin axis in degrees",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=60.0,
        help="camera distance in units of M",
    )
    parser.add_argument("--fov", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--backend", choices=("cpu", "gpu"), default="cpu"
    )
    parser.add_argument("--output", type=str, default="shadow.png")
    return parser.parse_args()


def main() -> None:
    """Trace one ray per pixel and write the shadow image."""
    args = parse_args()
    xp = get_xp(args.backend)
    spacetime = KerrSpacetime(mass=1.0, spin=args.spin)
    camera = PinholeCamera.from_orbit(
        distance=args.distance,
        inclination_degrees=args.inclination,
        fov_degrees=args.fov,
        width=args.width,
        height=args.height,
    )

    positions = xp.asarray(camera.ray_origins())
    directions = xp.asarray(camera.ray_directions())
    momenta = null_momentum_from_velocity(
        spacetime, positions, directions, time_orientation="past"
    )
    state0 = build_state(positions, momenta)

    print(
        f"Tracing {args.width * args.height} rays, spin={args.spin}, "
        f"inclination={args.inclination} deg, backend={args.backend}"
    )
    start = time.perf_counter()
    result = trace_rays(
        spacetime,
        state0,
        escape_radius=2.0 * args.distance,
        max_steps=args.max_steps,
        rtol=args.rtol,
    )
    elapsed = time.perf_counter() - start

    status_host = result.status
    if args.backend == "gpu":
        status_host = status_host.get()
    n = status_host.shape[0]
    captured = int((status_host == int(RayStatus.CAPTURED)).sum())
    escaped = int((status_host == int(RayStatus.ESCAPED)).sum())
    other = n - captured - escaped
    print(
        f"Done in {elapsed:.1f} s: {captured} captured "
        f"({100.0 * captured / n:.1f} percent), {escaped} escaped, "
        f"{other} other"
    )

    image = render_image(spacetime, result, args.width, args.height)
    save_png(image, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
