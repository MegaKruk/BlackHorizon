"""Tidal disruption event demonstration.

Disrupts a star at pericenter around a spinning black hole, evolves the
debris as independent Kerr geodesics (the validated Stage 1 machinery),
and writes top-down snapshot images of the stream plus the analytic
fallback light curve as a CSV. Pure NumPy plus the imaging module; no
OpenGL required.

Run:
    python -m blackhorizon.examples.tde_demo --output-dir tde_frames
"""

from __future__ import annotations

import argparse
import pathlib

import numpy

from ..dynamics.tde import fallback_rate, generate_debris_stream
from ..geodesics import geodesic_rhs
from ..imaging import save_png
from ..integrators import rk4_step
from ..kerr import KerrSpacetime


def render_topdown(
    positions: numpy.ndarray,
    energies: numpy.ndarray,
    spacetime: KerrSpacetime,
    extent: float,
    size: int = 720,
) -> numpy.ndarray:
    """Scatter debris onto a top-down image of the equatorial region.

    Bound particles render warm (orange), unbound ones cool (blue); the
    horizon is drawn as a filled dark disk with a thin marker ring.

    Args:
        positions: Particle positions, shape (n, 3).
        energies: Conserved specific energies, shape (n,).
        spacetime: The Kerr spacetime, for the horizon radius.
        extent: Half-width of the imaged square in units of M.
        size: Image size in pixels.

    Returns:
        RGB image array of shape (size, size, 3), dtype uint8.
    """
    image = numpy.zeros((size, size, 3), dtype=numpy.float64)
    scale = size / (2.0 * extent)
    px = ((positions[:, 0] + extent) * scale).astype(int)
    py = ((extent - positions[:, 1]) * scale).astype(int)
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    bound = energies < 1.0
    warm = numpy.array([1.0, 0.55, 0.15])
    cool = numpy.array([0.25, 0.5, 1.0])
    for x, y, is_bound in zip(px[inside], py[inside], bound[inside]):
        image[y, x] += warm if is_bound else cool

    # Soft brightness roll-off where particles pile up.
    image = 255.0 * image / (1.0 + image)

    yy, xx = numpy.mgrid[0:size, 0:size]
    radius = numpy.hypot(xx - size / 2.0, yy - size / 2.0) / scale
    horizon = spacetime.outer_horizon_radius
    image[radius <= horizon] = 8.0
    ring = numpy.abs(radius - horizon) < (1.5 / scale)
    image[ring] = numpy.array([90.0, 90.0, 110.0])
    return image.astype(numpy.uint8)


def main() -> None:
    """Run the disruption, integrate the debris, write frames and curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin", type=float, default=0.9)
    parser.add_argument("--star-mass", type=float, default=1e-6)
    parser.add_argument("--star-radius", type=float, default=0.5)
    parser.add_argument("--particles", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps-per-frame", type=int, default=800)
    parser.add_argument("--proper-time-step", type=float, default=0.6)
    parser.add_argument("--output-dir", type=str, default="tde_frames")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spacetime = KerrSpacetime(mass=1.0, spin=args.spin)
    stream = generate_debris_stream(
        spacetime,
        star_mass=args.star_mass,
        star_radius=args.star_radius,
        n_particles=args.particles,
    )
    print(
        f"disrupted star at r_t = {stream.tidal_radius:.1f} M, "
        f"{stream.bound_fraction * 100:.1f} pct of debris bound"
    )

    state = stream.states
    step = numpy.full((state.shape[0],), args.proper_time_step)
    extent = stream.tidal_radius * 3.0

    def rhs(batch: numpy.ndarray) -> numpy.ndarray:
        return geodesic_rhs(spacetime, batch)

    for frame in range(args.frames):
        image = render_topdown(
            state[:, 1:4], stream.specific_energies, spacetime, extent
        )
        path = output_dir / f"tde_{frame:03d}.png"
        save_png(image, str(path))
        radii = spacetime.kerr_schild_radius(
            state[:, 1], state[:, 2], state[:, 3]
        )
        print(
            f"frame {frame}: median r = {float(numpy.median(radii)):.1f} M "
            f"-> {path}"
        )
        for _ in range(args.steps_per_frame):
            state = rk4_step(rhs, state, step)

    # Analytic fallback light curve for the bound debris.
    times = numpy.geomspace(1.0, 300.0, 200)
    rate = fallback_rate(
        times,
        peak_time=1.0,
        disrupted_mass=args.star_mass * stream.bound_fraction,
    )
    curve_path = output_dir / "fallback_curve.csv"
    with open(curve_path, "w") as handle:
        handle.write("time,mdot\n")
        for t, m in zip(times, rate):
            handle.write(f"{t:.6e},{m:.6e}\n")
    print(f"fallback curve (t^-5/3) -> {curve_path}")


if __name__ == "__main__":
    main()
