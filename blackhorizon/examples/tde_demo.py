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


def _splat_kernel(radius: int = 3, sigma: float = 1.3) -> numpy.ndarray:
    """Small Gaussian footprint used to draw each particle."""
    x = numpy.arange(-radius, radius + 1)
    xx, yy = numpy.meshgrid(x, x)
    kernel = numpy.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.max()


def deposit_splats(
    trail: numpy.ndarray,
    positions: numpy.ndarray,
    energies: numpy.ndarray,
    extent: float,
) -> None:
    """Add particle splats onto the trail buffer in place.

    Bound particles deposit warm (orange), unbound cool (blue), as
    additive Gaussian footprints. Called several times per output frame
    so moving particles paint continuous streaks instead of dots.

    Args:
        trail: Accumulation buffer, shape (size, size, 3), modified.
        positions: Particle positions, shape (n, 3).
        energies: Conserved specific energies, shape (n,).
        extent: Half-width of the imaged square in units of M.
    """
    size = trail.shape[0]
    kernel = _splat_kernel()
    pad = kernel.shape[0] // 2
    scale = size / (2.0 * extent)
    px = ((positions[:, 0] + extent) * scale).astype(int)
    py = ((extent - positions[:, 1]) * scale).astype(int)
    inside = (
        (px >= pad)
        & (px < size - pad)
        & (py >= pad)
        & (py < size - pad)
    )
    bound = energies < 1.0
    warm = numpy.array([1.0, 0.55, 0.15])
    cool = numpy.array([0.3, 0.55, 1.0])
    for x, y, is_bound in zip(px[inside], py[inside], bound[inside]):
        tint = warm if is_bound else cool
        trail[
            y - pad : y + pad + 1, x - pad : x + pad + 1
        ] += kernel[:, :, None] * tint[None, None, :]


def compose_frame(
    trail: numpy.ndarray,
    spacetime: KerrSpacetime,
    extent: float,
) -> numpy.ndarray:
    """Develop the trail buffer into a display frame.

    Brightness auto-normalizes to the buffer's 99.5th percentile under
    an asinh stretch, astronomy's usual trick for high dynamic range;
    the horizon is a filled dark disk with a marker ring.

    Args:
        trail: Accumulation buffer, shape (size, size, 3).
        spacetime: The Kerr spacetime, for the horizon radius.
        extent: Half-width of the imaged square in units of M.

    Returns:
        RGB image array of shape (size, size, 3), dtype uint8.
    """
    size = trail.shape[0]
    scale = size / (2.0 * extent)
    peak = numpy.percentile(trail.max(axis=-1), 99.5)
    normalized = trail / max(peak, 1e-9)
    stretched = numpy.arcsinh(6.0 * normalized) / numpy.arcsinh(6.0)
    image = 255.0 * numpy.clip(stretched, 0.0, 1.0)

    yy, xx = numpy.mgrid[0:size, 0:size]
    radius = numpy.hypot(xx - size / 2.0, yy - size / 2.0) / scale
    horizon = spacetime.outer_horizon_radius
    image[radius <= horizon] = 8.0
    ring = numpy.abs(radius - horizon) < max(1.5 / scale, 1.0)
    image[ring] = numpy.array([110.0, 110.0, 135.0])
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
    parser.add_argument(
        "--deposits-per-frame",
        type=int,
        default=8,
        help="trail deposits per frame; more paints smoother streaks",
    )
    parser.add_argument("--proper-time-step", type=float, default=0.6)
    parser.add_argument(
        "--extent",
        type=float,
        default=5.0,
        help="half-width of the view in tidal radii; fixed so the "
        "trail buffer and any encoded video stay geometrically "
        "consistent",
    )
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

    def rhs(batch: numpy.ndarray) -> numpy.ndarray:
        return geodesic_rhs(spacetime, batch)

    size = 720
    trail = numpy.zeros((size, size, 3), dtype=numpy.float64)
    extent = args.extent * stream.tidal_radius
    frame_decay = 0.80
    deposits = max(1, args.deposits_per_frame)
    decay_per_deposit = frame_decay ** (1.0 / deposits)
    chunk = max(1, args.steps_per_frame // deposits)

    for frame in range(args.frames):
        for _ in range(deposits):
            trail *= decay_per_deposit
            deposit_splats(
                trail, state[:, 1:4], stream.specific_energies, extent
            )
            for _ in range(chunk):
                state = rk4_step(rhs, state, step)
        image = compose_frame(trail, spacetime, extent)
        path = output_dir / f"tde_{frame:03d}.png"
        save_png(image, str(path))
        radii_now = spacetime.kerr_schild_radius(
            state[:, 1], state[:, 2], state[:, 3]
        )
        print(
            f"frame {frame}: median r = "
            f"{float(numpy.median(radii_now)):.1f} M, "
            f"view half-width {extent:.0f} M -> {path}"
        )

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
