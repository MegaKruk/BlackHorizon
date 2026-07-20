"""Programmable doomed-observer plunge rendering.

Builds a radial infall worldline (a timelike geodesic) from an energy
preset, samples it evenly in proper time, constructs the infalling
tetrad at each sample, and renders maximum-fidelity frames through the
horizon down to the terminal radius. The presets:

- rain: free fall from rest at infinity, E = 1 (the Doran observer);
  interior transit takes 4M/3 of proper time (Lewis and Kwan 2007).
- maximal: the E = 0 trajectory, momentarily at rest at the horizon;
  the longest possible interior life, tau = pi M.
- fast: E > 1, falling with excess energy; the plunge is shorter.

Example:

    python -m blackhorizon.offline.infall --frames 10 --width 960 \
        --height 600 --look outward --output-dir plunge_frames
"""

from __future__ import annotations

import argparse
import pathlib

import numpy

from ..emission.bluesheet import blueshift_amplification, display_amplification
from ..frames import build_tetrad, raise_index
from ..geodesics import build_state, geodesic_rhs, hamiltonian
from ..imaging import save_png
from ..integrators import rk4_step
from ..kerr import KerrSpacetime
from ..realtime.fly_camera import FlyCamera
from .post import develop
from .render import OfflineSettings, render_hdr


def radial_momentum(
    spacetime: KerrSpacetime,
    position: numpy.ndarray,
    energy: float,
) -> numpy.ndarray:
    """Covariant momentum of a radial infaller with conserved energy.

    Solves the timelike mass shell for p = (-E, w l) with l the
    Kerr-Schild null vector (the E-parameterized Doran family; rain is
    E = 1) and selects the future-directed root; for E = 0 the state
    exists only inside the horizon (it is momentarily at rest at the
    horizon itself). At zero spin l is the radial direction.

    Args:
        spacetime: The Kerr spacetime.
        position: Spatial position, shape (3,).
        energy: Conserved energy E >= 0 per unit mass.

    Returns:
        Covariant momentum, shape (1, 4).

    Raises:
        ValueError: If no future-directed timelike root exists there.
    """
    pos = numpy.asarray(position, dtype=float)[None, :]
    p_t = -float(energy)
    h_field = float(
        spacetime.geometry(pos[:, 0], pos[:, 1], pos[:, 2]).h[0]
    )
    # Mass shell: (1 - 2H) w^2 + 4 H p_t w - (1 + 2H) p_t^2 + 1 = 0.
    coefficients = [
        1.0 - 2.0 * h_field,
        4.0 * h_field * p_t,
        1.0 - (1.0 + 2.0 * h_field) * p_t**2,
    ]
    l_vector = spacetime.geometry(
        pos[:, 0], pos[:, 1], pos[:, 2]
    ).l[0]
    for w in sorted(numpy.roots(coefficients).real, reverse=True):
        momentum = numpy.array([[p_t, 0.0, 0.0, 0.0]])
        momentum[0, 1:4] = w * l_vector
        state = build_state(pos, momentum)
        if abs(float(hamiltonian(spacetime, state)[0]) + 0.5) > 1e-9:
            continue
        velocity = raise_index(spacetime, pos, momentum)
        radial = float(
            numpy.dot(velocity[0, 1:4], pos[0])
        ) / numpy.linalg.norm(pos[0])
        if float(velocity[0, 0]) > 0.0 and radial < 0.0:
            return momentum
    raise ValueError(
        "no future-directed infalling root; for E = 0 start inside "
        "the horizon"
    )


def build_plunge(
    spacetime: KerrSpacetime,
    start_radius: float,
    energy: float,
    stop_radius: float,
    step: float = 5e-4,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Integrate the plunge worldline and return (taus, states).

    Args:
        spacetime: The Kerr spacetime.
        start_radius: Starting radius on the +x axis, units of M.
        energy: Conserved energy of the infall.
        stop_radius: Terminal radius near the singularity.
        step: Proper-time integration step.

    Returns:
        Tuple of proper times, shape (k,), and geodesic states,
        shape (k, 8), sampled every integration step.
    """
    position = numpy.array([start_radius, 0.0, 0.0])
    state = build_state(
        position[None, :], radial_momentum(spacetime, position, energy)
    )

    def rhs(batch):
        return geodesic_rhs(spacetime, batch)

    taus = [0.0]
    states = [state[0].copy()]
    h = numpy.array([step])
    for _ in range(2000000):
        radius = float(
            spacetime.kerr_schild_radius(
                state[:, 1], state[:, 2], state[:, 3]
            )[0]
        )
        step_now = float(min(step, max(0.04 * radius**1.5, 1e-6)))
        h[0] = step_now
        state = rk4_step(rhs, state, h)
        # Hold the timelike mass shell against stiff-field drift and
        # stop at the chart boundary (the ring plane or a Cauchy
        # horizon exit) instead of integrating garbage.
        value = float(hamiltonian(spacetime, state)[0])
        if not numpy.isfinite(state).all() or not numpy.isfinite(
            value
        ) or value >= -1e-6:
            break
        if abs(value + 0.5) > 1e-9:
            state[:, 4:8] *= numpy.sqrt(0.5 / (-value))
        taus.append(taus[-1] + step_now)
        states.append(state[0].copy())
        radius = float(
            spacetime.kerr_schild_radius(
                state[:, 1], state[:, 2], state[:, 3]
            )[0]
        )
        if radius <= stop_radius:
            break
    return numpy.asarray(taus), numpy.stack(states)


def plunge_tetrad(
    spacetime: KerrSpacetime, state: numpy.ndarray, look: str
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Camera tetrad at a worldline sample.

    Args:
        spacetime: The Kerr spacetime.
        state: Geodesic state, shape (8,).
        look: View mode: outward, inward, or side.

    Returns:
        Tuple (position, tetrad) with the tetrad from build_tetrad;
        spatial legs are re-orthonormalized each frame against fixed
        seeds, which keeps radial plunges roll-stable.
    """
    position = state[1:4]
    velocity = raise_index(
        spacetime, position[None, :], state[None, 4:8]
    )[0]
    outward = position / max(numpy.linalg.norm(position), 1e-12)
    up_seed = numpy.array([0.0, 0.0, 1.0])
    if abs(float(numpy.dot(outward, up_seed))) > 0.98:
        up_seed = numpy.array([0.0, 1.0, 0.0])
    if look == "outward":
        forward = outward
    elif look == "inward":
        forward = -outward
    else:
        forward = numpy.cross(up_seed, outward)
        forward /= max(numpy.linalg.norm(forward), 1e-12)
    tetrad = build_tetrad(
        spacetime, position, velocity, forward, up_seed
    )
    return position, tetrad


def render_plunge(args: argparse.Namespace) -> None:
    """Render evenly spaced proper-time frames along the plunge."""
    spacetime = KerrSpacetime(mass=1.0, spin=args.spin)
    energies = {"rain": 1.0, "maximal": 0.0, "fast": args.energy}
    energy = energies[args.preset]
    start_radius = args.start_radius
    if args.preset == "maximal":
        start_radius = min(
            start_radius, spacetime.outer_horizon_radius * 0.995
        )
    stop = args.stop
    inner = float(spacetime.inner_horizon_radius)
    if args.journey == "idealized" and inner > 0.0:
        print(
            "idealized journey: continuing past the Cauchy horizon "
            "into exact vacuum Kerr (labeled non-physical for real "
            "black holes)"
        )
    elif inner > 0.0 and stop < inner * 1.02:
        # Realistic terminal surface for spinning holes: the Cauchy
        # horizon, where the blue sheet ends any physical infall
        # (Poisson-Israel mass inflation; Dafermos-Luk
        # arXiv:1710.01722). Idealized continuation is Stage 6.
        stop = inner * 1.02
        print(
            f"spin {args.spin}: journey terminates at the Cauchy "
            f"horizon, stop raised to {stop:.4f} M"
        )
    taus, states = build_plunge(
        spacetime, start_radius, energy, stop
    )
    total = float(taus[-1])
    print(
        f"plunge: E = {energy}, r0 = {start_radius:.3f} M, "
        f"terminal tau = {total:.4f} M "
        f"({len(taus)} worldline samples)"
    )

    settings = OfflineSettings(
        spin=args.spin,
        supersample=args.supersample,
        disk_enabled=not args.no_disk,
        backend=args.backend,
        fov_degrees=args.fov,
    )
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_taus = numpy.linspace(0.0, total * args.span, args.frames)
    last = args.frames if args.end_frame is None else args.end_frame
    for index, tau in enumerate(frame_taus):
        if index < args.start_frame or index >= last:
            continue
        sample = int(numpy.searchsorted(taus, tau))
        sample = min(sample, len(taus) - 1)
        state = states[sample]
        position, tetrad = plunge_tetrad(spacetime, state, args.look)
        radius = float(
            spacetime.kerr_schild_radius(
                state[None, 1], state[None, 2], state[None, 3]
            )[0]
        )
        camera = FlyCamera(position=position.copy(), yaw=0.0, pitch=0.0)
        hdr = render_hdr(
            camera,
            args.width,
            args.height,
            settings,
            progress=False,
            camera_tetrad=tetrad,
            interior_stop=stop,
        )
        # Photographic adaptation: near the blue sheet any camera
        # stops down; the console still reports the true state.
        exposure = args.exposure
        amplification = 1.0
        if args.journey == "realistic" and inner > 0.0:
            amplification = float(
                blueshift_amplification(
                    spacetime, numpy.array([radius])
                )[0]
            )
            adapted = float(
                display_amplification(
                    numpy.array([amplification])
                )[0]
            )
            exposure = args.exposure / adapted**1.5
        image = develop(hdr, exposure=exposure)
        path = output_dir / f"plunge_{index:03d}.png"
        save_png(image, str(path))
        side = "inside" if radius <= spacetime.outer_horizon_radius else "outside"
        sheet_note = (
            f", blue sheet B = {amplification:.1f}"
            if amplification > 1.001
            else ""
        )
        print(
            f"frame {index}: tau = {tau:.3f} M, r = {radius:.3f} M "
            f"({side}){sheet_note}, remaining {total - tau:.3f} M "
            f"-> {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a doomed-observer plunge at maximum fidelity"
    )
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument(
        "--preset",
        choices=("rain", "maximal", "fast"),
        default="rain",
        help="infall energy: rain E=1, maximal E=0, fast E>1",
    )
    parser.add_argument("--energy", type=float, default=1.5)
    parser.add_argument("--start-radius", type=float, default=6.0)
    parser.add_argument("--stop", type=float, default=0.02)
    parser.add_argument(
        "--journey",
        choices=("realistic", "idealized"),
        default="realistic",
        help="at the Cauchy horizon of a spinning hole: terminate "
        "(realistic, the blue sheet) or continue into exact vacuum "
        "Kerr (idealized)",
    )
    parser.add_argument(
        "--span",
        type=float,
        default=0.985,
        help="fraction of the total plunge covered by the frames",
    )
    parser.add_argument(
        "--look", choices=("outward", "inward", "side"), default="outward"
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="first frame index to render (resume support)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="one past the last frame index to render",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--exposure", type=float, default=1.4)
    parser.add_argument("--no-disk", action="store_true")
    parser.add_argument(
        "--backend", choices=("cpu", "gpu"), default="cpu"
    )
    parser.add_argument("--output-dir", default="plunge_frames")
    render_plunge(parser.parse_args())


if __name__ == "__main__":
    main()
