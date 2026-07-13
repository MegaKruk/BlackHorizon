"""Offline maximum-fidelity renderer.

Renders single frames with the full Stage 1 physics: float64 adaptive
Dormand-Prince geodesic integration, disk crossings localized by
bisection along the trajectory (not linear interpolation), a
tolerance-tightening second pass for rays that exhaust their step budget
near the photon shell, subpixel supersampling, and linear HDR output
developed through the post module (bloom, ACES, sRGB). This is the
authoritative image; the real-time GLSL renderer approximates it.

The tracing loop is array-module generic, so the gpu backend (CuPy)
accelerates it; emission runs on the CPU where the hit count is small.

Run:
    python -m blackhorizon.offline.render --spin 0.9 --output frame.png
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy

from ..backend import get_xp, to_numpy, xp_of
from ..emission.blackbody import blackbody_lut
from ..emission.novikov_thorne import disk_inner_radius, temperature_lut
from ..emission.redshift import redshift_factor, static_observer_lapse
from ..geodesics import build_state, geodesic_rhs, null_momentum_from_velocity
from ..imaging import save_png
from ..integrators import (
    dormand_prince_step,
    error_ratio,
    rk4_step,
    step_factor,
)
from ..kerr import KerrSpacetime
from ..realtime.fly_camera import FlyCamera
from ..tracer import RayStatus
from .post import develop


@dataclass(frozen=True)
class OfflineSettings:
    """Physics and quality parameters of an offline render.

    Attributes:
        spin: Black hole spin a/M.
        fov_degrees: Vertical field of view.
        supersample: Subpixel grid side; 2 traces 4 rays per pixel.
        rtol: Adaptive tolerance of the first pass.
        refine_rtol: Tolerance of the photon-shell refinement pass.
        max_steps: Step budget of the first pass; the refinement pass
            quadruples it.
        max_step: Upper bound on the affine step.
        capture_margin: Capture at r <= r_plus (1 + margin).
        disk_enabled: Whether the Novikov-Thorne disk is rendered.
        disk_outer_radius: Outer disk radius in units of M (clamped
            above the ISCO).
        disk_temperature: Peak effective temperature in Kelvin.
        disk_detail: Strength of the procedural streak modulation,
            matching the real-time shader formula.
        tile_rays: Rays traced per tile, bounding memory.
        backend: Array backend, "cpu" or "gpu".
    """

    spin: float = 0.9
    fov_degrees: float = 70.0
    supersample: int = 2
    rtol: float = 1e-9
    refine_rtol: float = 1e-11
    max_steps: int = 12000
    max_step: float = 2.0
    capture_margin: float = 1e-3
    disk_enabled: bool = True
    disk_outer_radius: float = 18.0
    disk_temperature: float = 6500.0
    disk_detail: float = 1.0
    tile_rays: int = 120000
    backend: str = "cpu"


@dataclass
class _TraceOutput:
    """Per-ray results of one traced tile."""

    status: numpy.ndarray
    escape_directions: numpy.ndarray
    hit_positions: numpy.ndarray = field(default=None)
    hit_momenta: numpy.ndarray = field(default=None)
    saturated: numpy.ndarray = field(default=None)


def subpixel_directions(
    camera: FlyCamera,
    width: int,
    height: int,
    fov_degrees: float,
    supersample: int,
) -> numpy.ndarray:
    """View directions for every subpixel sample, shape (n, 3).

    Samples sit on a regular supersample x supersample grid inside each
    pixel, ordered row-major by pixel then by subpixel, so averaging
    groups of supersample^2 consecutive rays downsamples the image.
    """
    forward, right, up = camera.basis()
    tan_half = math.tan(math.radians(fov_degrees) / 2.0)
    aspect = width / height
    ss = supersample
    offsets = (numpy.arange(ss) + 0.5) / ss
    px = (
        numpy.arange(width)[:, None] + offsets[None, :]
    ).reshape(-1)
    py = (
        numpy.arange(height)[:, None] + offsets[None, :]
    ).reshape(-1)
    u = px / width * 2.0 - 1.0
    v = 1.0 - py / height * 2.0
    # Order: pixel row, pixel column, subpixel row, subpixel column.
    u_grid = numpy.tile(
        u.reshape(width, ss)[None, :, None, :], (height, 1, ss, 1)
    )
    v_grid = numpy.tile(
        v.reshape(height, ss)[:, None, :, None], (1, width, 1, ss)
    )
    directions = (
        forward[None, :]
        + u_grid.reshape(-1)[:, None] * tan_half * right[None, :]
        + v_grid.reshape(-1)[:, None] * (tan_half / aspect) * up[None, :]
    )
    return directions / numpy.linalg.norm(
        directions, axis=-1, keepdims=True
    )


def _flat_directions(
    spacetime: KerrSpacetime, positions, momenta
):
    """Asymptotic flat-space direction of escaped rays.

    Mirrors the shader: dx = p - 2 H lp l with lp = -p_t + l . p and
    the traced p_t = +1.
    """
    xp = xp_of(positions)
    geo = spacetime.geometry(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    lp = -1.0 + xp.sum(geo.l * momenta, axis=-1)
    directions = momenta - 2.0 * (geo.h * lp)[:, None] * geo.l
    norm = xp.sqrt(xp.sum(directions * directions, axis=-1))
    return directions / norm[:, None]


def _trace_tile(
    spacetime: KerrSpacetime,
    state0,
    escape_radius: float,
    settings: OfflineSettings,
    disk_radii: tuple[float, float] | None,
    rtol: float,
    max_steps: int,
) -> _TraceOutput:
    """Adaptively trace one tile of rays with disk crossings refined.

    The loop mirrors tracer.trace_rays and adds, after each accepted
    step, equatorial crossing detection with bisection refinement along
    a Runge-Kutta re-integration of the crossing step, localizing the
    hit far below the accepted step size.
    """
    xp = xp_of(state0)
    n = state0.shape[0]
    capture_radius = spacetime.outer_horizon_radius * (
        1.0 + settings.capture_margin
    )

    def rhs(batch):
        return geodesic_rhs(spacetime, batch)

    states = state0.copy()
    status = xp.full((n,), int(RayStatus.IN_FLIGHT), dtype=xp.int32)
    steps = xp.zeros((n,), dtype=xp.int64)
    h = xp.full((n,), 0.1, dtype=state0.dtype)
    hit_positions = xp.zeros((n, 3), dtype=state0.dtype)
    hit_momenta = xp.zeros((n, 3), dtype=state0.dtype)

    iterations = 0
    while iterations < 4 * max_steps:
        idx = xp.nonzero(status == int(RayStatus.IN_FLIGHT))[0]
        if idx.size == 0:
            break
        iterations += 1

        y = states[idx]
        radius = spacetime.kerr_schild_radius(y[:, 1], y[:, 2], y[:, 3])
        captured = radius <= capture_radius
        escaped = ~captured & (radius >= escape_radius)
        exhausted = steps[idx] >= max_steps
        status[idx[captured]] = int(RayStatus.CAPTURED)
        status[idx[escaped]] = int(RayStatus.ESCAPED)
        status[idx[exhausted & ~captured & ~escaped]] = int(
            RayStatus.MAX_STEPS
        )
        alive = ~(captured | escaped | exhausted)
        idx = idx[alive]
        if idx.size == 0:
            continue

        y = states[idx]
        h_a = h[idx]
        y_new, err = dormand_prince_step(rhs, y, h_a)
        ratio = error_ratio(y, y_new, err, rtol, rtol * 1e-3)
        accept = ratio <= 1.0

        if disk_radii is not None:
            crossed = accept & (y[:, 3] * y_new[:, 3] < 0.0)
            if bool(crossed.any()):
                hits, positions, momenta = _refine_crossings(
                    spacetime,
                    rhs,
                    y[crossed],
                    h_a[crossed],
                    disk_radii,
                )
                targets = idx[xp.nonzero(crossed)[0][hits]]
                status[targets] = int(RayStatus.DISK)
                hit_positions[targets] = positions
                hit_momenta[targets] = momenta

        still = status[idx] == int(RayStatus.IN_FLIGHT)
        y = xp.where(accept[:, None], y_new, y)
        states[idx] = xp.where(still[:, None], y, states[idx])
        steps[idx] = steps[idx] + (accept & still).astype(steps.dtype)
        h[idx] = xp.clip(h_a * step_factor(ratio), 0.0, settings.max_step)

    saturated = (status == int(RayStatus.MAX_STEPS)) | (
        steps >= int(0.85 * max_steps)
    )

    escape_mask = status == int(RayStatus.ESCAPED)
    directions = xp.zeros((n, 3), dtype=state0.dtype)
    if bool(escape_mask.any()):
        directions[escape_mask] = _flat_directions(
            spacetime,
            states[escape_mask][:, 1:4],
            states[escape_mask][:, 5:8],
        )

    return _TraceOutput(
        status=to_numpy(status),
        escape_directions=to_numpy(directions),
        hit_positions=to_numpy(hit_positions),
        hit_momenta=to_numpy(hit_momenta),
        saturated=to_numpy(saturated),
    )


def _refine_crossings(
    spacetime: KerrSpacetime,
    rhs,
    y0,
    h_full,
    disk_radii: tuple[float, float],
):
    """Bisect the crossing step to localize equatorial hits.

    From the pre-step states, a Runge-Kutta substep of fractional size
    is a fourth-order sample of the same trajectory; forty bisection
    iterations pin the crossing to a 1e-12 fraction of the step. Rays
    whose refined radius falls outside the disk annulus are reported as
    misses (the ray continues past the gap or beyond the rim).

    Returns:
        Tuple (hits, positions, momenta) in the caller's array module,
        where hits is a boolean mask over the input rays and the arrays
        cover only the hits.
    """
    xp = xp_of(y0)
    z0 = y0[:, 3]
    low = xp.zeros(y0.shape[0], dtype=y0.dtype)
    high = xp.ones(y0.shape[0], dtype=y0.dtype)
    for _ in range(40):
        mid = 0.5 * (low + high)
        y_mid = rk4_step(rhs, y0, h_full * mid)
        same_side = y_mid[:, 3] * z0 > 0.0
        low = xp.where(same_side, mid, low)
        high = xp.where(same_side, high, mid)
    y_cross = rk4_step(rhs, y0, h_full * 0.5 * (low + high))
    r_hit = spacetime.kerr_schild_radius(
        y_cross[:, 1], y_cross[:, 2], y_cross[:, 3]
    )
    hits = (r_hit >= disk_radii[0]) & (r_hit <= disk_radii[1])
    # Stay in the tracing array module (NumPy or CuPy); the caller
    # indexes device arrays with this mask.
    return hits, y_cross[hits][:, 1:4], y_cross[hits][:, 5:8]


def _starfield_hdr(directions: numpy.ndarray) -> numpy.ndarray:
    """Deterministic HDR starfield sampled by view direction.

    Two hash-based layers of point stars with blackbody-like tints on a
    latitude-longitude cell grid, plus a faint warm band around the
    equatorial plane of the sky. Values can exceed one so bright stars
    bloom in post.
    """
    phi = numpy.arctan2(directions[:, 1], directions[:, 0])
    theta = numpy.arccos(numpy.clip(directions[:, 2], -1.0, 1.0))
    color = numpy.zeros((directions.shape[0], 3))

    def hash01(a, b, salt):
        v = numpy.sin(a * 127.1 + b * 311.7 + salt * 74.7) * 43758.5453
        return v - numpy.floor(v)

    for cells, density, brightness in ((160, 0.10, 3.0), (420, 0.05, 0.9)):
        cu = phi / (2.0 * numpy.pi) * cells
        cv = theta / numpy.pi * cells
        iu, iv = numpy.floor(cu), numpy.floor(cv)
        present = hash01(iu, iv, 1.0) < density
        su = iu + 0.15 + 0.7 * hash01(iu, iv, 2.0)
        sv = iv + 0.15 + 0.7 * hash01(iu, iv, 3.0)
        dist2 = (cu - su) ** 2 + (cv - sv) ** 2
        magnitude = hash01(iu, iv, 4.0)
        amplitude = brightness * (0.15 + magnitude**4)
        point = numpy.exp(-dist2 / 0.006)
        warmth = hash01(iu, iv, 5.0)
        tint = numpy.stack(
            [
                0.75 + 0.35 * warmth,
                0.85 + 0.10 * warmth,
                1.05 - 0.35 * warmth,
            ],
            axis=-1,
        )
        color += (present * amplitude * point)[:, None] * tint

    band = 0.035 * numpy.exp(-((directions[:, 2] / 0.30) ** 2))
    color += band[:, None] * numpy.array([1.0, 0.92, 0.80])
    return color


def _disk_radiance(
    spacetime: KerrSpacetime,
    positions: numpy.ndarray,
    momenta: numpy.ndarray,
    settings: OfflineSettings,
    disk_inner: float,
    disk_outer: float,
    temperature_table: numpy.ndarray,
    observer_lapse: float,
    bb_table: numpy.ndarray,
    bb_log_min: float,
    bb_log_max: float,
) -> numpy.ndarray:
    """Linear HDR radiance of disk hits, mirroring the shader model.

    A blackbody at T redshifts to a blackbody at g T, so evaluating the
    chromaticity at g T with a (g T)^4 brightness reproduces the exact
    g^4 bolometric scaling.
    """
    r_hit = to_numpy(
        spacetime.kerr_schild_radius(
            positions[:, 0], positions[:, 1], positions[:, 2]
        )
    )
    lut_radii = numpy.linspace(
        disk_inner, disk_outer, temperature_table.shape[0]
    )
    t_norm = numpy.interp(r_hit, lut_radii, temperature_table)

    momenta4 = numpy.concatenate(
        [numpy.ones((positions.shape[0], 1)), momenta], axis=1
    )
    shift = to_numpy(
        redshift_factor(spacetime, positions, momenta4, observer_lapse)
    )
    t_observed = numpy.maximum(
        shift * t_norm * settings.disk_temperature, 1.0
    )

    log_t = numpy.clip(numpy.log(t_observed), bb_log_min, bb_log_max)
    lut_u = (
        (log_t - bb_log_min)
        / (bb_log_max - bb_log_min)
        * (bb_table.shape[0] - 1)
    )
    tint = numpy.stack(
        [
            numpy.interp(
                lut_u, numpy.arange(bb_table.shape[0]), bb_table[:, c]
            )
            for c in range(3)
        ],
        axis=-1,
    )

    phi = numpy.arctan2(positions[:, 1], positions[:, 0])
    detail = 1.0 + settings.disk_detail * (
        0.18 * numpy.sin(9.0 * phi + 2.2 * r_hit)
        + 0.12 * numpy.sin(23.0 * phi - 5.0 * r_hit)
        + 0.15 * numpy.sin(3.5 * (r_hit - disk_inner))
    )
    brightness = (t_observed / 6500.0) ** 4 * numpy.maximum(detail, 0.2)
    return tint * brightness[:, None]


def render_hdr(
    camera: FlyCamera,
    width: int,
    height: int,
    settings: OfflineSettings,
    progress: bool = True,
) -> numpy.ndarray:
    """Render a linear HDR frame at maximum fidelity.

    Rays that exhaust the first-pass step budget (they wind near the
    photon shell where trajectories are exponentially sensitive) are
    retraced with hundredfold tighter tolerance and a quadrupled budget
    before being classified.

    Args:
        camera: The viewpoint.
        width: Image width in pixels.
        height: Image height in pixels.
        settings: Physics and quality parameters.
        progress: Print per-tile progress.

    Returns:
        Linear HDR image, shape (height, width, 3), float32.
    """
    spacetime = KerrSpacetime(mass=1.0, spin=settings.spin)
    xp = get_xp(settings.backend)
    disk_inner = disk_inner_radius(spacetime)
    disk_outer = max(settings.disk_outer_radius, disk_inner + 1.0)
    disk_radii = (
        (disk_inner, disk_outer) if settings.disk_enabled else None
    )
    temperature_table, _, _ = temperature_lut(
        settings.spin, disk_outer, size=2048
    )
    bb_table, bb_log_min, bb_log_max = blackbody_lut(size=1024)
    try:
        observer_lapse = static_observer_lapse(spacetime, camera.position)
    except ValueError:
        observer_lapse = 1.0
    escape_radius = 1.3 * max(camera.distance_from_origin, disk_outer)

    directions = subpixel_directions(
        camera, width, height, settings.fov_degrees, settings.supersample
    )
    n = directions.shape[0]
    radiance = numpy.zeros((n, 3), dtype=numpy.float64)
    start = time.perf_counter()

    for begin in range(0, n, settings.tile_rays):
        end = min(begin + settings.tile_rays, n)
        tile_directions = directions[begin:end]
        positions = numpy.tile(
            camera.position[None, :], (tile_directions.shape[0], 1)
        )
        momenta = null_momentum_from_velocity(
            spacetime, positions, tile_directions, time_orientation="past"
        )
        state0 = xp.asarray(build_state(positions, momenta))

        result = _trace_tile(
            spacetime,
            state0,
            escape_radius,
            settings,
            disk_radii,
            settings.rtol,
            settings.max_steps,
        )

        # Photon-shell refinement pass with tighter tolerance.
        retrace = result.saturated & (
            result.status != int(RayStatus.DISK)
        )
        if bool(retrace.any()):
            refined = _trace_tile(
                spacetime,
                state0[xp.asarray(numpy.nonzero(retrace)[0])],
                escape_radius,
                settings,
                disk_radii,
                settings.refine_rtol,
                settings.max_steps * 4,
            )
            for name in ("status", "escape_directions"):
                getattr(result, name)[retrace] = getattr(refined, name)
            result.hit_positions[retrace] = refined.hit_positions
            result.hit_momenta[retrace] = refined.hit_momenta

        tile_radiance = numpy.zeros(
            (tile_directions.shape[0], 3), dtype=numpy.float64
        )
        escaped = result.status == int(RayStatus.ESCAPED)
        if escaped.any():
            tile_radiance[escaped] = _starfield_hdr(
                result.escape_directions[escaped]
            )
        disk_hits = result.status == int(RayStatus.DISK)
        if disk_hits.any():
            tile_radiance[disk_hits] = _disk_radiance(
                spacetime,
                result.hit_positions[disk_hits],
                result.hit_momenta[disk_hits],
                settings,
                disk_inner,
                disk_outer,
                temperature_table,
                observer_lapse,
                bb_table,
                bb_log_min,
                bb_log_max,
            )
        radiance[begin:end] = tile_radiance
        if progress:
            elapsed = time.perf_counter() - start
            print(
                f"  traced {end}/{n} rays "
                f"({end / n * 100.0:.0f} pct, {elapsed:.0f} s)"
            )

    ss2 = settings.supersample * settings.supersample
    hdr = radiance.reshape(height, width, ss2, 3).mean(axis=2)
    return hdr.astype(numpy.float32)


def main() -> None:
    """Command line entry point for a single maximum-fidelity frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--distance", type=float, default=26.0)
    parser.add_argument("--inclination", type=float, default=82.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--no-disk", action="store_true")
    parser.add_argument("--disk-outer", type=float, default=18.0)
    parser.add_argument("--disk-temperature", type=float, default=6500.0)
    parser.add_argument("--disk-detail", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=1.4)
    parser.add_argument("--bloom-strength", type=float, default=0.35)
    parser.add_argument("--bloom-sigma", type=float, default=6.0)
    parser.add_argument("--bloom-threshold", type=float, default=1.0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--output", type=str, default="offline_frame.png")
    parser.add_argument(
        "--hdr-output",
        type=str,
        default="",
        help="optional .npy path for the linear HDR frame",
    )
    args = parser.parse_args()

    settings = OfflineSettings(
        spin=args.spin,
        fov_degrees=args.fov,
        supersample=args.supersample,
        rtol=args.rtol,
        max_steps=args.max_steps,
        disk_enabled=not args.no_disk,
        disk_outer_radius=args.disk_outer,
        disk_temperature=args.disk_temperature,
        disk_detail=args.disk_detail,
        backend=args.backend,
    )
    camera = FlyCamera.from_orbit(
        args.distance, args.inclination, args.azimuth
    )
    start = time.perf_counter()
    hdr = render_hdr(camera, args.width, args.height, settings)
    image = develop(
        hdr,
        exposure=args.exposure,
        bloom_threshold=args.bloom_threshold,
        bloom_strength=args.bloom_strength,
        bloom_sigma=args.bloom_sigma,
    )
    save_png(image, args.output)
    if args.hdr_output:
        numpy.save(args.hdr_output, hdr)
    print(
        f"Rendered {args.width}x{args.height} ss={args.supersample} "
        f"spin={args.spin} in {time.perf_counter() - start:.1f} s, "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
