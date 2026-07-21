"""Offline mass-inflation flythrough (Stage 7, charged-Vaidya/Ori).

Where Stage 6 ends the realistic journey at the blue sheet with a
whiteout, this renderer continues through it, using the spherical
charged-Vaidya/Ori surrogate (blackhorizon.vaidya) to model the
mass-inflation layer that Kerr's angular structure makes analytically
intractable. It is offline only and deliberately non-realtime: the
metric is time dependent, so photon energy is not conserved and every
ray integrates its own p_t through the evolving geometry.

The camera rides the ingoing shell-free geodesic (the surrogate's
rain analogue) from between the horizons, through the inflating shell,
to the weak null singularity at the render horizon. At each sampled
proper time it builds an orthonormal tetrad on the frozen slice and
traces past-directed rays with the full time-dependent flow, shading
escaped rays by the surrogate sky map and the inflating layer by the
locally measured Misner-Sharp mass. The result is a rendered passage
through the region a realistic infaller would traverse with finite
tidal distortion (Ori 1991; Marolf and Ori arXiv:1109.5139).
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy

from ..emission.bluesheet import SHEET_COLOR
from ..frames import build_tetrad, lower_index, raise_index
from ..geodesics import build_state
from ..imaging import save_png
from ..integrators import rk4_step
from ..vaidya import ChargedVaidyaSpacetime, vaidya_geodesic_rhs
from .post import develop
from .render import OfflineSettings


def rain_state(
    spacetime: ChargedVaidyaSpacetime,
    position: numpy.ndarray,
    time: float,
) -> numpy.ndarray:
    """Ingoing free-fall camera state on the frozen slice.

    Uses the surrogate's rain analogue: covariant p = (-1, w l) with
    w = -sqrt(2H)/(1 + sqrt(2H)) evaluated on the slice, the
    Painleve-Gullstrand infaller, regular through both horizons.
    """
    slice_geo = spacetime.frozen(time)
    geo = slice_geo.geometry(
        position[None, 0], position[None, 1], position[None, 2]
    )
    root = math.sqrt(max(2.0 * float(geo.h[0]), 0.0))
    w = -root / (1.0 + root)
    covariant = numpy.array(
        [[-1.0, w * geo.l[0, 0], w * geo.l[0, 1], w * geo.l[0, 2]]]
    )
    state = numpy.empty((1, 8))
    state[0, 0] = time
    state[0, 1:4] = position
    state[0, 4:8] = covariant[0]
    return state


def camera_four_velocity(
    spacetime: ChargedVaidyaSpacetime, state: numpy.ndarray
) -> numpy.ndarray:
    """Contravariant camera 4-velocity from the covariant state."""
    slice_geo = spacetime.frozen(float(state[0, 0]))
    return raise_index(
        slice_geo, state[:, 1:4], state[:, 4:8]
    )[0]


def build_worldline(
    spacetime: ChargedVaidyaSpacetime,
    start_radius: float,
    start_time: float,
    stop_radius: float,
    step: float = 2e-3,
    mass_threshold: float = 1e4,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Integrate the camera worldline through the inflation layer.

    Returns proper times and states, sampled every step, from the
    start radius to the render horizon just outside r = stop_radius.
    """
    position = numpy.array([start_radius, 0.0, 0.0])
    state = rain_state(spacetime, position, start_time)

    def rhs(batch):
        return vaidya_geodesic_rhs(spacetime, batch)

    r_minus = spacetime.inner_horizon_radius
    tau = 0.0
    taus = [0.0]
    states = [state[0].copy()]
    record_every = 8
    for iteration in range(120000):
        radius = float(
            spacetime.kerr_schild_radius(
                state[:, 1], state[:, 2], state[:, 3]
            )[0]
        )
        # Proximity-floored step: shrink toward the terminal surface
        # but never below a floor, so the camera crosses the short
        # remaining proper-time interval in bounded work.
        proximity = max(radius - r_minus, 5e-3)
        step_now = float(min(step, max(0.05 * proximity, 2e-3)))
        state = rk4_step(rhs, state, numpy.array([step_now]))
        if not numpy.isfinite(state).all():
            break
        tau += step_now
        radius = float(
            spacetime.kerr_schild_radius(
                state[:, 1], state[:, 2], state[:, 3]
            )[0]
        )
        v_here = float(state[0, 0]) + radius
        mass_here = float(
            spacetime.misner_sharp_mass(
                numpy.array([v_here]), numpy.array([radius])
            )[0]
        )
        reached = (
            radius <= stop_radius or mass_here >= mass_threshold
        )
        if iteration % record_every == 0 or reached:
            taus.append(tau)
            states.append(state[0].copy())
        if reached:
            break
    return numpy.asarray(taus), numpy.stack(states)


def _surrogate_sky(directions: numpy.ndarray) -> numpy.ndarray:
    """Faint deterministic starfield for escaped rays, shape (n, 3)."""
    def hash01(a, b, salt):
        h = numpy.sin(a * 12.9898 + b * 78.233 + salt * 37.719)
        return (h * 43758.5453) % 1.0

    theta = numpy.arccos(numpy.clip(directions[:, 2], -1.0, 1.0))
    phi = numpy.arctan2(directions[:, 1], directions[:, 0])
    color = numpy.zeros((directions.shape[0], 3))
    for cells, brightness in ((38.0, 1.6), (85.0, 0.8)):
        iu = numpy.floor(phi / (2 * numpy.pi) * cells)
        iv = numpy.floor(theta / numpy.pi * cells)
        fu = (phi / (2 * numpy.pi) * cells) % 1.0
        fv = (theta / numpy.pi * cells) % 1.0
        cx, cy = hash01(iu, iv, 1.0), hash01(iu, iv, 2.0)
        dist2 = (fu - cx) ** 2 + (fv - cy) ** 2
        mag = hash01(iu, iv, 4.0)
        amp = brightness * (0.2 + mag**4) * numpy.exp(-dist2 / 0.004)
        warmth = hash01(iu, iv, 5.0)
        tint = numpy.stack(
            [0.6 + 0.4 * warmth, 0.6 + 0.2 * warmth, 1.0 - 0.3 * warmth],
            axis=-1,
        )
        color += amp[:, None] * tint
    return color


def _trace_vaidya_tile(
    spacetime: ChargedVaidyaSpacetime,
    state0: numpy.ndarray,
    escape_radius: float,
    stop_radius: float,
    max_steps: int = 900,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Trace past-directed rays through the time-dependent surrogate.

    Fixed-step RK4 with a horizon-crossing-safe scale; returns the
    per-ray status (0 escaped, 1 terminal surface, 2 budget), the
    escape directions, the local Misner-Sharp mass at the terminal
    point, and the advanced time there (for shading the inflating
    layer).
    """
    n = state0.shape[0]
    state = state0.copy()
    status = numpy.full(n, -1, dtype=int)
    escape_dir = numpy.zeros((n, 3))
    end_mass = numpy.zeros(n)
    active = numpy.ones(n, dtype=bool)
    r_plus = spacetime.outer_horizon_radius

    def rhs(batch):
        return vaidya_geodesic_rhs(spacetime, batch)

    for _ in range(max_steps):
        if not active.any():
            break
        idx = numpy.nonzero(active)[0]
        sub = state[idx]
        radius = spacetime.kerr_schild_radius(
            sub[:, 1], sub[:, 2], sub[:, 3]
        )
        momentum2 = numpy.sum(sub[:, 5:8] ** 2, axis=-1)
        captured = (radius <= stop_radius) | (
            (radius <= stop_radius * 1.08) & (momentum2 > 400.0)
        ) | (momentum2 > 1.0e6)
        escaped = ~captured & (radius >= escape_radius)
        if captured.any():
            hit = idx[captured]
            v = sub[captured, 0] + radius[captured]
            end_mass[hit] = spacetime.misner_sharp_mass(
                v, radius[captured]
            )
            status[hit] = 1
        if escaped.any():
            esc = idx[escaped]
            geo = spacetime.frozen(0.0).geometry(
                sub[escaped, 1], sub[escaped, 2], sub[escaped, 3]
            )
            p_s = sub[escaped, 5:8]
            lp = -sub[escaped, 4] + numpy.sum(geo.l * p_s, axis=-1)
            direction = p_s - 2.0 * geo.h[:, None] * lp[:, None] * geo.l
            escape_dir[esc] = direction / numpy.linalg.norm(
                direction, axis=-1, keepdims=True
            )
            status[esc] = 0
        active[idx[captured | escaped]] = False
        idx = numpy.nonzero(active)[0]
        if idx.size == 0:
            break
        sub = state[idx]
        radius = spacetime.kerr_schild_radius(
            sub[:, 1], sub[:, 2], sub[:, 3]
        )
        scale = numpy.where(
            radius > r_plus, radius - r_plus, 0.5 * radius
        )
        h = numpy.clip(0.15 * scale, 3e-3, 0.08)
        state[idx] = rk4_step(rhs, sub, h)
    status[status == -1] = 2
    v_end = state[:, 0] + spacetime.kerr_schild_radius(
        state[:, 1], state[:, 2], state[:, 3]
    )
    return status, escape_dir, end_mass, v_end


def render_layer_frame(
    spacetime: ChargedVaidyaSpacetime,
    state: numpy.ndarray,
    width: int,
    height: int,
    settings: OfflineSettings,
    look: str = "outward",
) -> numpy.ndarray:
    """Render one flythrough frame from a worldline state."""
    position = state[1:4]
    velocity = camera_four_velocity(spacetime, state[None, :])
    slice_geo = spacetime.frozen(float(state[0]))
    outward = position / max(numpy.linalg.norm(position), 1e-12)
    up_seed = numpy.array([0.0, 0.0, 1.0])
    if abs(float(numpy.dot(outward, up_seed))) > 0.98:
        up_seed = numpy.array([0.0, 1.0, 0.0])
    if look == "inward":
        forward = -outward
    elif look == "side":
        forward = numpy.cross(up_seed, outward)
        forward /= max(numpy.linalg.norm(forward), 1e-12)
    else:
        forward = outward
    try:
        norm = float(numpy.linalg.norm(velocity))
        if not numpy.isfinite(velocity).all() or norm > 1e4:
            raise ValueError("diverged worldline sample")
        tetrad = build_tetrad(
            slice_geo, position, velocity, forward, up_seed
        )
    except ValueError:
        # Rain fallback on the frozen slice keeps the frame renderable
        # even if the worldline state has left the valid region.
        geo = slice_geo.geometry(
            position[None, 0], position[None, 1], position[None, 2]
        )
        root = math.sqrt(max(2.0 * float(geo.h[0]), 0.0))
        w = -root / (1.0 + root)
        cov = numpy.array(
            [[-1.0, w * geo.l[0, 0], w * geo.l[0, 1], w * geo.l[0, 2]]]
        )
        rain_v = raise_index(slice_geo, position[None, :], cov)[0]
        tetrad = build_tetrad(
            slice_geo, position, rain_v, forward, up_seed
        )

    tan_half = math.tan(math.radians(settings.fov_degrees) / 2.0)
    aspect = width / height
    ss = settings.supersample
    offsets = (numpy.arange(ss) + 0.5) / ss
    xs = ((numpy.arange(width)[:, None] + offsets[None, :]).reshape(-1))
    ys = ((numpy.arange(height)[:, None] + offsets[None, :]).reshape(-1))
    u = xs / width * 2.0 - 1.0
    v = 1.0 - ys / height * 2.0
    u_grid = numpy.tile(
        u.reshape(width, ss)[None, :, None, :], (height, 1, ss, 1)
    ).reshape(-1)
    v_grid = numpy.tile(
        v.reshape(height, ss)[:, None, :, None], (1, width, 1, ss)
    ).reshape(-1)
    local = numpy.stack(
        [numpy.ones_like(u_grid), u_grid * tan_half, v_grid * tan_half / aspect],
        axis=-1,
    )
    local /= numpy.linalg.norm(local, axis=-1, keepdims=True)

    spatial = (
        local[:, 0:1] * tetrad[1]
        + local[:, 1:2] * tetrad[2]
        + local[:, 2:3] * tetrad[3]
    )
    physical = tetrad[0][None, :] - spatial
    positions = numpy.tile(position[None, :], (local.shape[0], 1))
    momenta = lower_index(slice_geo, positions, -physical)
    time_column = numpy.full((local.shape[0], 1), float(state[0]))
    state0 = numpy.concatenate([time_column, positions, momenta], axis=1)

    r_here = float(numpy.linalg.norm(position))
    escape_radius = max(12.0, 2.5 * r_here)
    status, escape_dir, end_mass, _ = _trace_vaidya_tile(
        spacetime, state0, escape_radius, spacetime.inner_horizon_radius * 1.001
    )

    radiance = numpy.zeros((local.shape[0], 3), dtype=numpy.float32)
    escaped = status == 0
    if escaped.any():
        radiance[escaped] = _surrogate_sky(escape_dir[escaped])
    layer = status == 1
    if layer.any():
        # The inflating layer glows by the local Misner-Sharp mass at
        # the ray's terminal point relative to the background M: a
        # blue-white membrane whose brightness tracks the quasilocal
        # mass jump across the shell (the visible signature of mass
        # inflation).
        excess = numpy.maximum(
            end_mass[layer] - spacetime.mass, 0.0
        ) / spacetime.mass
        glow = 0.35 + 1.2 * numpy.log1p(excess) ** 2
        radiance[layer] = glow[:, None] * numpy.array(
            SHEET_COLOR, dtype=numpy.float32
        )

    image = radiance.reshape(height, width, ss * ss, 3).mean(axis=2)
    return image


def render_flythrough(args: argparse.Namespace) -> None:
    """Render the mass-inflation flythrough sequence."""
    spacetime = ChargedVaidyaSpacetime(
        charge=args.charge,
        tail_mass=args.tail_mass,
        shell_energy=args.shell_energy,
        shell_start_v=args.shell_v,
        shell_start_radius=args.shell_radius,
    )
    print(
        f"charged-Vaidya surrogate: q = {args.charge} "
        f"(models Kerr a = {args.charge}), horizons "
        f"{spacetime.inner_horizon_radius:.4f} .. "
        f"{spacetime.outer_horizon_radius:.4f}, "
        f"kappa_minus = {spacetime.inner_surface_gravity:.4f}"
    )
    taus, states = build_worldline(
        spacetime,
        args.start_radius,
        args.start_time,
        spacetime.inner_horizon_radius * 1.001,
    )
    total = float(taus[-1])
    print(
        f"worldline: start r = {args.start_radius} M at v = "
        f"{args.start_time}, terminal tau = {total:.4f} M "
        f"({len(taus)} samples)"
    )

    settings = OfflineSettings(
        spin=0.0,
        supersample=args.supersample,
        disk_enabled=False,
        fov_degrees=args.fov,
    )
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_taus = numpy.linspace(0.0, total * args.span, args.frames)
    for index, tau in enumerate(frame_taus):
        sample = min(int(numpy.searchsorted(taus, tau)), len(taus) - 1)
        state = states[sample]
        radius = float(numpy.linalg.norm(state[1:4]))
        v = float(state[0]) + radius
        mass = float(
            spacetime.misner_sharp_mass(
                numpy.array([v]), numpy.array([radius])
            )[0]
        )
        hdr = render_layer_frame(
            spacetime, state, args.width, args.height, settings, args.look
        )
        image = develop(hdr, exposure=args.exposure)
        path = output_dir / f"inflation_{index:03d}.png"
        save_png(image, str(path))
        print(
            f"frame {index}: tau = {tau:.3f} M, r = {radius:.4f} M, "
            f"v = {v:.2f}, Misner-Sharp mass = {mass:.3e} M -> {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a mass-inflation flythrough (Ori surrogate)"
    )
    parser.add_argument("--charge", type=float, default=0.9)
    parser.add_argument("--tail-mass", type=float, default=0.01)
    parser.add_argument("--shell-energy", type=float, default=1e-3)
    parser.add_argument("--shell-v", type=float, default=6.0)
    parser.add_argument("--shell-radius", type=float, default=1.2)
    parser.add_argument("--start-radius", type=float, default=1.3)
    parser.add_argument("--start-time", type=float, default=6.0)
    parser.add_argument("--span", type=float, default=0.995)
    parser.add_argument(
        "--look", choices=("outward", "inward", "side"), default="side"
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--exposure", type=float, default=1.5)
    parser.add_argument("--output-dir", default="inflation_frames")
    render_flythrough(parser.parse_args())


if __name__ == "__main__":
    main()
