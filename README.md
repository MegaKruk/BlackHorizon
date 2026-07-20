# Black Horizon

Interactive black hole and orbital dynamics simulator. Stage 1: the core
general-relativistic engine.

This stage implements full Kerr (spinning black hole) physics: geodesics are
integrated in horizon-penetrating Cartesian Kerr-Schild coordinates from a
Hamiltonian formulation, with an adaptive Dormand-Prince 5(4) integrator,
on the CPU (NumPy) or the GPU (CuPy) from the same code. It is validated
end to end against exact general-relativistic results and ships a shadow
ray tracer that renders the gravitationally lensed view of the hole.

The full design, physics references and roadmap live in docs/DESIGN.md.
Stage 2 (real-time ModernGL rendering), Stage 3 (accretion disk, tidal
disruption events, post-Newtonian N-body) and Stage 4 (offline maximum
fidelity renderer) build on this package.

## Project layout

    blackhorizon/
        realtime/        Stage 2: GLSL tracer, engine, fly camera, app
        backend.py       NumPy/CuPy dispatch
        kerr.py          Kerr spacetime, Kerr-Schild geometry, analytic radii
        geodesics.py     Equations of motion, initial conditions, invariants
        integrators.py   Batched RK4 and Dormand-Prince 5(4)
        tracer.py        Adaptive ray propagation with termination
        camera.py        Pinhole camera
        imaging.py       Shadow image shading and PNG output
        examples/
            render_shadow.py
            benchmark.py
    tests/               Physics validation suite
    docs/DESIGN.md       Design document (implementation anchor)

## Stage 2: real-time interactive mode

The lensed black hole view is traced live, one geodesic per pixel, inside
a GLSL fragment shader (a transcription of the validated Stage 1 physics;
accuracy is enforced by tests against the adaptive tracer). Install the
real-time extras and launch:

    pip install -e ".[dev,realtime]"
    python -m blackhorizon.realtime.app --spin 0.9

The Novikov-Thorne accretion disk is on by default; the settings panel
exposes its outer radius, peak temperature, exposure, tone detail, and
an on/off toggle. Colors are physical: a blackbody palette shifted by
the covariant redshift factor, so the approaching side beams hot and
bright while the receding side dims and reddens.

Controls: WASD and Q/E to fly, right mouse drag to look, left shift to
boost, R resets the camera, 1/2/3/4 select quality presets, B toggles the
background, [ and ] nudge the spin, F12 saves a screenshot, Escape quits.
If imgui-bundle is installed a settings panel exposes spin, field of
view, step budget, and resolution scale; without it the keyboard
controls above still work (start with --no-ui to force that mode).

Useful flags: --quality low|medium|high|ultra, --distance, --inclination,
--width, --height, --no-vsync (uncapped frame rate for benchmarking).

To verify the GL path without opening a window, or to take stills:

    python -m blackhorizon.realtime.headless --spin 0.9 --quality ultra --output frame.png

Quality presets trade step budget for frame rate; their physical accuracy
is quantified in docs/DESIGN.md section 9.

Wayland note: the app automatically makes pyGLFW share imgui-bundle's
GLFW library (see realtime/glfw_compat.py), because pyGLFW's Wayland-only
build otherwise collides with imgui-bundle's X11-linked native module at
import time. The window then runs through XWayland. To override the
library choice, set PYGLFW_LIBRARY yourself; if the window fails to open,
GLFW_PLATFORM=x11 remains a useful fallback.

## Setup (Manjaro Linux, PyCharm venv)

With your venv activated in the project folder:

    pip install -e ".[dev]"

Optional GPU acceleration on the RTX 3070 (requires only the NVIDIA
proprietary driver, which on Manjaro is installed via mhwd; verify with
nvidia-smi):

    pip install cupy-cuda12x

## Run the validation suite

    pytest

The suite checks, among other things:

- metric identities and the Schwarzschild limit of the Kerr-Schild form
- analytic gradients against finite differences
- conservation of E, L_z and the Hamiltonian through strong-field flybys
- a photon riding the unstable circular orbit at r = 3M
- a massive particle on a circular orbit at r = 6M around an a = 0.9 hole
- the Schwarzschild shadow boundary at b = 3 sqrt(3) M within 0.5 percent
- the Kerr (a = 0.9) prograde and retrograde shadow boundaries against
  Bardeen's analytic critical impact parameters within 0.5 percent

## Render a black hole shadow

    python -m blackhorizon.examples.render_shadow --spin 0.9 --output shadow.png

Useful flags: --inclination (degrees from the spin axis), --distance,
--fov, --width, --height, --backend gpu. Captured photons form the black
shadow; escaped photons sample a celestial checkerboard so the lensing and
the frame-dragging asymmetry of the shadow are visible. Expect roughly one
to two minutes on the CPU at 480x360; the GPU backend is much faster.

## Benchmark

    python -m blackhorizon.examples.benchmark

Reports nanoseconds per ray-step for the raw geodesic integration on each
available backend. Reference measurement of this Stage 1 array-based code
on a container-class CPU: about 1700 ns per ray-step (float64). The GPU
path and the planned raw-kernel port (see docs/DESIGN.md, section 3) exist
precisely to close the gap to the sub-nanosecond figures published for
native CUDA tracers.

## Units and conventions

Geometric units G = c = 1 with the black hole mass M = 1 setting the
length scale. Metric signature (-, +, +, +). State arrays are (n, 8):
(t, x, y, z, p_t, p_x, p_y, p_z) with covariant momenta. Spin a is in
[-M, M]; the spin axis is +z.


## Dynamics modules (Stage 3)

- blackhorizon.dynamics.peters: gravitational-wave inspiral tracks
  (Peters 1964), for example integrate_inspiral(1.0, 1.0, 60.0, 0.3).
- blackhorizon.dynamics.pn_nbody: Newtonian, 1PN Einstein-Infeld-
  Hoffmann, and 2.5PN radiation-reaction accelerations with an RK4
  stepper, for comparable-mass systems outside the strong field.
- blackhorizon.dynamics.tde: tidal disruption prescriptions and a
  geodesic debris stream generator.

Tidal disruption demo (writes top-down debris frames and the analytic
fallback curve, no OpenGL needed):

    python -m blackhorizon.examples.tde_demo --output-dir tde_frames


## Offline rendering and video (Stage 4)

Maximum-fidelity still (float64 adaptive tracing, bisection-refined
disk crossings, photon-shell refinement, supersampling, bloom):

    python -m blackhorizon.offline.render --spin 0.9 --width 1920 \
        --height 1200 --supersample 2 --backend gpu --output hero.png

Use --backend cpu without CuPy; --hdr-output saves the linear frame as
.npy for regrading. Orbit video through the real-time engine (install
the video extra first: pip install -e ".[video]"):

    python -m blackhorizon.offline.video --seconds 10 --fps 30 \
        --width 1280 --height 720 --quality ultra --output orbit.mp4

Encode existing PNG frames (for example the TDE demo output):

    python -m blackhorizon.offline.video --encode-frames tde_frames \
        --fps 12 --output tde.mp4

For long orbital evolutions, integrators.implicit_midpoint_step is a
symplectic alternative to RK4 with bounded long-term energy error.

## Stage 5: flying into the black hole

In the realtime app, fly across the event horizon (WASDQE toward the
hole; the panel shows your r in units of M). At the crossing the
camera becomes a doomed observer: a timelike worldline integrated in
proper time on the rain (free-fall) trajectory. Inside:

- The panel turns red: INSIDE THE HORIZON, with your current r, a
  proper-time countdown to the singularity, and a time-scale slider.
- WASDQE fire thrusters (exact local-frame boosts). No burn can
  increase r inside; burns generally shorten your remaining proper
  time, and braking toward the E = 0 trajectory lengthens it, up to
  the absolute maximum of pi M from the horizon.
- Right-drag still looks around: more than half the sky shows the
  outside universe just after crossing, and the accretion disk seen
  transversely turns blue-white from blueshift.
- The light-cone overlay draws the river-model inflow: inside, both
  cone edges point inward.
- The reset camera button (or R) is the only way out.
- In realistic mode the approach to the inner horizon of a spinning
  hole now renders the blue sheet quantitatively: the whole external
  sky and the disk blueshift and brighten as B = x_match/x while the
  panel reads out the amplification, the wall ahead glows blue-white,
  and the final moments white out as the radiation bath overwhelms
  any adapted exposure. Idealized mode stays dark there, as eternal
  vacuum Kerr should.
- For a spinning hole the panel offers two journey modes past the
  inner (Cauchy) horizon. Realistic (default): the journey ends
  there, where the infinitely blueshifted blue sheet terminates any
  physical infall (mass inflation). Idealized Kerr: continue into
  the inner region of exact eternal vacuum Kerr, clearly labeled as
  non-physical for real black holes; inside the Cauchy horizon your
  radius can legally increase again, the outside universe stays
  visible through the horizon above you, and the journey ends at the
  ring plane or when an outgoing branch reaches the edge of the
  coordinate chart. Switchable mid-flight; --journey selects it in
  the offline plunge CLI. Standing assumptions either way: you are
  an indestructible test observer (radiation and tides are already
  waived outside), and the idealized region reflects the eternal
  vacuum solution, not a prediction about real holes. The
  Schwarzschild plunge (--spin 0) always runs the full distance to
  the central singularity.

Offline, render a maximum-fidelity plunge:

    python -m blackhorizon.offline.infall --frames 10 --width 960 \
        --height 600 --look side --start-radius 8 \
        --output-dir plunge_frames
    python -m blackhorizon.offline.video --encode-frames plunge_frames \
        --fps 8 --output plunge.mp4

Presets: --preset rain (E = 1, default), maximal (E = 0, the longest
possible interior life), fast --energy 1.5. Views: --look outward
(the shrinking sky), inward (the darkness ahead), side (disk and sky).
