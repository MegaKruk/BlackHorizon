# Black Horizon - Design Document

Status: living document. This is the implementation anchor derived from the
deep research report (2026-07). All stages must stay consistent with it.

## 1. Project goals

Black Horizon is an interactive black hole and orbital dynamics simulator:

- Physics as detailed and current as published science allows: full general
  relativity with Kerr (spinning) black holes, not Newtonian approximations.
- GPU-accelerated: heavy lifting on an NVIDIA RTX 3070 (8 GB VRAM, Ampere,
  20.31 TFLOPS FP32, FP64 at 1/64 FP32 rate), host OS Manjaro Linux.
- Language: Python 3.14, open source libraries only.
- Two modes:
  1. Real-time interactive: fly camera, live parameter panels (spin, mass,
     disk, quality), adjustable fidelity and resolution.
  2. Offline simulation: configure once, render frame by frame at maximum
     fidelity, export video.
- Clean, modular, SRP-conforming code. No emojis, no non-keyboard characters.

## 2. Physics foundation (with primary references)

### 2.1 Kerr spacetime

Geometric units G = c = 1, black hole mass M, spin a with |a| <= M.

Boyer-Lindquist (BL) quantities (used for setup and analytics only, never for
integration near the horizon):

- Sigma = r^2 + a^2 cos^2(theta), Delta = r^2 - 2 M r + a^2
- Horizons: r_plus_minus = M +- sqrt(M^2 - a^2)
- Ergosphere outer boundary: r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))
- ISCO (Bardeen, Press and Teukolsky 1972, ApJ 178, 347), chi = a/M:
  Z1 = 1 + (1 - chi^2)^(1/3) * ((1 + chi)^(1/3) + (1 - chi)^(1/3))
  Z2 = sqrt(3 chi^2 + Z1^2)
  r_isco / M = 3 + Z2 -+ sqrt((3 - Z1)(3 + Z1 + 2 Z2))
  (minus sign: prograde). Limits: 6M at chi = 0, 1M prograde and 9M
  retrograde at chi = 1.
- Equatorial circular photon orbits (Bardeen 1973):
  r_ph = 2 M (1 + cos((2/3) arccos(-+ a / M))), prograde uses -a.
  Limits: 3M at a = 0; 1M prograde and 4M retrograde at a = M.
- Circular equatorial orbit angular velocity (prograde):
  Omega = sqrt(M) / (r^(3/2) + a sqrt(M))
- Conserved quantities along geodesics: E = -p_t, L_z = p_phi, the Carter
  constant Q (Carter 1968), and the Hamiltonian itself.

### 2.2 Coordinates: Cartesian Kerr-Schild (the key algorithmic decision)

BL coordinates are singular at the horizon and cause spurious swirling and
integrator blow-ups. All numerical integration is done in horizon-penetrating
Cartesian Kerr-Schild (KS) coordinates (as in GRay2, RAPTOR):

  g_mu_nu = eta_mu_nu + 2 H l_mu l_nu
  g^mu^nu = eta^mu^nu - 2 H l^mu l^nu        (exact, since l is null)

with eta = diag(-1, 1, 1, 1) and

  H = M r^3 / (r^4 + a^2 z^2)
  l_mu = (1, (r x + a y) / (r^2 + a^2), (r y - a x) / (r^2 + a^2), z / r)

where the KS radius r solves r^4 - (x^2 + y^2 + z^2 - a^2) r^2 - a^2 z^2 = 0:

  r^2 = ((rho^2 - a^2) + sqrt((rho^2 - a^2)^2 + 4 a^2 z^2)) / 2,
  rho^2 = x^2 + y^2 + z^2.

l_mu is null with respect to both eta and g. The metric is regular across the
horizon; the only true singularity is the ring (r = 0, z = 0).

### 2.3 Equations of motion: Hamiltonian formulation

H_ham(x, p) = (1/2) g^mu^nu p_mu p_nu
            = (1/2) eta^mu^nu p_mu p_nu - H (l^mu p_mu)^2

with l^mu p_mu = -p_t + l_x p_x + l_y p_y + l_z p_z. Hamilton's equations
(8 coupled first-order ODEs), writing lp = l^mu p_mu:

  dt/dlam   = -p_t + 2 H lp
  dx_i/dlam = p_i - 2 H lp l_i
  dp_t/dlam = 0                       (stationarity: E conserved exactly)
  dp_i/dlam = (dH/dx_i) lp^2 + 2 H lp sum_j p_j (dl_j/dx_i)

Analytic gradients (validated against finite differences in tests):
with S = sqrt((rho^2 - a^2)^2 + 4 a^2 z^2), Qd = r^4 + a^2 z^2:

  dr/dx = x r / S,  dr/dy = y r / S,  dr/dz = z (r^2 + a^2) / (r S)
  dH/dr = M r^2 (3 a^2 z^2 - r^4) / Qd^2
  dH/dz|_explicit = -2 M a^2 z r^3 / Qd^2
  dl_x/dx_i, dl_y/dx_i from the quotient rule on (r x + a y)/(r^2 + a^2)
  and (r y - a x)/(r^2 + a^2); dl_z/dx_i = (delta_iz - l_z dr/dx_i)/r.

Null rays: g^mu^nu p_mu p_nu = 0. Timelike: = -mu^2 (H_ham = -mu^2 / 2).

Initial conditions: choose a contravariant velocity direction k = (k_t, s)
and solve the quadratic g_mu_nu k^mu k^nu = 0 (null) for k_t, picking the
future or past root; then p_mu = g_mu_nu k^nu. Backward ray tracing from the
camera uses the past-directed root so that Kerr frame-dragging asymmetry has
the physically correct orientation in images.

### 2.4 Integrators

- Interactive and Stage 1: adaptive embedded Runge-Kutta, Dormand-Prince
  5(4), per-ray adaptive steps, error-controlled. RK4 fixed step kept for
  benchmarking. Rationale: accurate and efficient for short null geodesic
  traces (Wu et al. 2021 series).
- Offline maximum fidelity (Stage 4): explicit time-transformed symplectic
  integrators (Wu, Wang, Sun, Liu 2020-2021, ApJ; adaptive variant ApJS
  2024) for long timelike integrations where RK4 shows secular energy drift.
- Safety: cap steps per ray (GRay caps at about 2e6), terminate on horizon
  crossing (r <= r_plus (1 + margin)) and escape (r >= r_escape), shrink
  steps adaptively near the photon shell, guard the ring singularity.

### 2.5 Accretion disk (Stage 3)

Novikov-Thorne relativistic thin disk (Novikov and Thorne 1973; Page and
Thorne 1974): flux F(r) with zero-torque inner boundary at the ISCO,
T(r) = (F / sigma_SB)^(1/4). Observed intensity I_obs = g^3 I_em (bolometric
g^4) with the total redshift factor g = nu_obs / nu_em from the photon
momentum dotted into the emitter 4-velocity. Blackbody-to-RGB color mapping
plus Doppler beaming (Luminet 1979 asymmetry). Higher-order photon rings
arise naturally from the ray tracer. Reference appearance: EHT M87* (42
micro-arcsec ring) and Sgr A* (51.8 micro-arcsec ring). Visual style target:
DNGR / Interstellar paper (James, von Tunzelmann, Franklin, Thorne 2015,
CQG 32, 065001) but keeping the frequency shifts the movie omitted.

### 2.6 Tidal disruption events (Stage 3)

Tidal radius r_t ~= R_star (M_BH / M_star)^(1/3) (Hills 1975; Rees 1988).
Hills mass about 1e8 solar masses for a Sun-like star. Fallback rate
t^(-5/3) for full disruptions (Rees 1988), steepening to t^(-9/4) for
partial disruptions (Coughlin and Nixon 2019; Miles, Coughlin and Nixon
2020). Implementation tiers: (a) analytic light-curve prescription,
(b) test-particle debris cluster (1e3 to 1e5 particles on geodesics with an
energy spread). Full SPH is out of scope.

### 2.7 Multibody dynamics (Stage 3)

Test particles around one hole: Kerr geodesics directly. Comparable-mass
bodies: post-Newtonian N-body, Einstein-Infeld-Hoffmann equations with
1PN, 2PN and 2.5PN radiation reaction (REBOUNDx gr_full as the reference
implementation). Gravitational-wave inspiral via Peters 1964 (Phys. Rev.
136, B1224):

  T_circ = 5 c^5 a^4 / (256 G^3 m1 m2 (m1 + m2))
  da/dt and de/dt with the standard eccentricity enhancement factors.

PN is invalid in the strong field very near the horizon; switch to geodesics
there.

## 3. Technical stack decisions

- Python 3.14, standard GIL build (the GPU provides the parallelism; avoids
  the C-extension free-threading minefield). Verify with
  sys._is_gil_enabled() if experimenting with 3.14t.
- Physics arrays: NumPy (CPU reference and tests) and CuPy (GPU), written
  backend-agnostic so the same code runs on both. CuPy 14.x ships Python
  3.14 wheels; install cupy-cuda12x (only the NVIDIA driver is required).
- Hot-path escalation path (decision trigger): if the batched array
  implementation is not fast enough for real time, port the RHS + integrator
  loop into a single CuPy RawKernel (CUDA C) or a GLSL compute shader. The
  physics module is structured so the kernel is a direct transcription.
- Rendering (Stage 2): ModernGL + moderngl-window, imgui-bundle for panels,
  instanced sprites for particles, full-screen shader for the lensed view.
  CUDA-GL interop via mapped pixel/vertex buffers; fallback is pure-GLSL
  tracing if interop is unstable.
- Offline mode (Stage 4): progressive supersampling, FP64 promotion for rays
  near the photon shell, motion blur, bloom, PNG/EXR frames, video via
  imageio-ffmpeg or PyAV.
- Precision: FP64 default on CPU; FP32 allowed on GPU for interactive mode
  (5x to 28x faster on consumer GPUs), FP64 selectively where shadow-edge
  artifacts appear.

## 4. Performance expectations

Published GPU tracers reach about 1 ns or less per photon per RK step on
2013-2016 hardware (Odyssey, Pu et al. 2016, ApJ 820, 105; GRay, Chan et al.
2013). Extrapolated to the RTX 3070: sub-nanosecond per step in FP32 with a
native kernel. Batched CuPy array code will be slower (kernel-launch bound)
but still far above CPU. Render time ~= rays x steps/ray x ns/step.
Measure, do not assume: the Stage 1 benchmark script reports ns per
ray-step on both backends.

## 5. Architecture and module layout (SRP)

blackhorizon/
  backend.py      Array-module dispatch (NumPy or CuPy), transfers.
  kerr.py         KerrSpacetime: KS geometry, metric, analytic gradients,
                  horizons, ISCO, photon orbits. Pure functions of position.
  geodesics.py    Hamiltonian RHS, initial-condition builders (null and
                  timelike), conserved quantities, coordinate velocity.
  integrators.py  Generic batched steppers: RK4 fixed, Dormand-Prince 5(4)
                  with per-sample error estimate; dense integrate helper.
  tracer.py       Ray batch propagation: adaptive control, termination
                  (captured, escaped, max steps, failed), TraceResult.
  camera.py       PinholeCamera: pixel grid to world-space directions.
  imaging.py      Shading traced rays into an image (shadow plus celestial
                  sphere checkerboard), PNG output.
  examples/
    render_shadow.py   CLI: render the lensed shadow image.
    benchmark.py       CLI: ns per ray-step on CPU and GPU.
tests/            Physics validation suite (see section 6).
docs/DESIGN.md    This document.

Later stages add: emission/ (disk models, redshift, color), dynamics/
(particles, PN N-body, TDE), render/ (ModernGL pipelines), ui/ (imgui),
offline/ (frame scheduler, encoder). Every metric and integrator sits behind
the same call signatures so new spacetimes and schemes drop in.

## 6. Stage 1 validation criteria (all must pass)

1. Metric identities: g g_inverse = identity; l null in eta and g; KS radius
   correct on axis and in the equatorial plane; a = 0 reduces to
   Schwarzschild in KS form.
2. Analytic gradients of H and l match high-order finite differences.
3. Conservation: E exactly, L_z and H_ham to integrator tolerance over
   strong-field flybys and bound orbits.
4. Schwarzschild shadow: critical impact parameter within 1 percent of
   b_c = 3 sqrt(3) M = 5.196 M, measured end to end by capture/escape
   bisection from a distant observer.
5. Photon sphere: tangential photon at r = 3M stays on the unstable circular
   orbit for multiple orbits.
6. ISCO formula reproduces 6M (a = 0), 1M (a = M prograde), 9M (a = M
   retrograde); a circular massive-particle orbit at moderate radius stays
   circular over multiple orbits.
7. Kerr asymmetry: for a = 0.9 the prograde capture boundary sits at
   smaller |b| than Schwarzschild, the retrograde boundary at larger |b|.
8. Benchmark numbers recorded for CPU (and GPU when available).

## 7. Roadmap

- Stage 1 (this stage): core engine, validation, benchmark, shadow imager.
- Stage 2: ModernGL real-time mode, fly camera, imgui panels, quality
  settings. Threshold: 30 fps at 1080p, a = 0.9, on the RTX 3070.
- Stage 3: Novikov-Thorne disk with g-factor coloring, photon rings, TDE
  prescription plus debris particles, PN N-body, Peters inspiral.
- Stage 4: offline maximum-fidelity renderer and video export.

## 8. Risks and mitigations

- Coordinate singularities: only integrate in KS; guard the ring
  singularity with epsilon clamps; capture at the horizon.
- Unbounded orbiting near the photon shell: step caps and adaptive control.
- FP32 artifacts near the shadow edge: per-ray FP64 promotion (Stage 4).
- 8 GB VRAM: tile or stream large supersampled buffers.
- Python 3.14 ecosystem gaps: CuPy is the committed GPU dependency; Numba,
  Taichi and PyTorch are deliberately not load-bearing.
- Model limits to document in-app: Novikov-Thorne invalid inside the ISCO,
  TDE prescriptions approximate, PN invalid in the strong field.

## 9. Stage 2 addendum: real-time mode (implemented)

Decisions taken, superseding open options above:

- Tracing path: pure GLSL fragment shader (option chosen over CUDA-GL
  interop). The shader in realtime/shaders/kerr_tracer.frag is a direct
  transcription of the Stage 1 physics; a float64 NumPy mirror of its
  exact algorithm lives in realtime/reference.py and is held to the
  Stage 1 adaptive tracer by tests/test_realtime_reference.py.
- Integration: fixed-order RK4 with the step heuristic
  h = clip(step_scale * (r - r_plus), min_step, max_step), plus a
  momentum-aware clamp h <= 1 / max(1, |p|). Two lessons learned and
  encoded in tests: (a) past-directed rays entering the shadow asymptote
  to the horizon with exponentially diverging blueshift, and the term
  dp ~ grad_h (l.p)^2 makes fixed steps unstable there, so a diverging
  momentum (|p| > momentum_bailout, default 1e3) is itself the capture
  criterion; (b) the ring singularity guard must survive squaring in
  float arithmetic (guard 1e-30, not 1e-300).
- Measured preset accuracy against the Stage 1 adaptive tracer
  (equatorial fans, both sides, distance 30 M): low/medium/high/ultra all
  give 100 percent capture classification agreement at a = 0 and
  a = 0.9. At a = 0.998: high and ultra remain exact, medium shows about
  1 percent prograde boundary error, low about 6 percent.
- Float32 sufficiency: the compiled GLSL agrees with the float64
  reference on 100 percent of pixels in the headless test scenes
  (tests/test_realtime_gl.py); per-ray FP64 promotion remains a Stage 4
  concern only.
- Stack: moderngl (context and offscreen target), glfw (window and
  input), imgui-bundle (optional settings panel; the app degrades to
  keyboard-only controls without it). Internal render resolution is
  decoupled from the window via resolution_scale and blitted.

Recorded benchmarks (RTX 3070, Manjaro, Python 3.14.5, 2026-07): batched
CuPy array path 242.2 ns per ray-step float64, 77.7 ns float32,
CPU NumPy 1376 ns; this confirmed the design prediction that the array
path is kernel-launch bound and motivated the pure-shader tracer.
Container reference: llvmpipe software GL renders 640x480 at the high
preset in about 0.9 s, so the RTX 3070 has orders of magnitude of
headroom for 1080p at 30 to 60 fps.


## Stage 3 addendum: emission and dynamics (implemented)

Stage 2 field acceptance: 97 fps at approximately 1281 x 872 on the
ultra preset with the settings panel active (RTX 3070, driver 595.71.05),
comfortably above the 30 fps design threshold.

### What Stage 3 delivered

Emission (blackhorizon/emission/):
- novikov_thorne.py: closed-form Page and Thorne (1974) flux with the
  x^3 - 3x + 2a = 0 root decomposition; the a = 0 removable singularity
  (a root at x = 0) is handled by dropping the vanishing-coefficient
  term. Zero-torque inner boundary at the ISCO; plunging-region emission
  is neglected (documented limitation). Validated: F = 0 at the ISCO,
  temperature peak at 9.55 M for a = 0 moving inward with spin, and the
  large-radius Shakura-Sunyaev T ~ r^(-3/4) slope.
- blackbody.py: Planck spectrum integrated against the CIE 1931 color
  matching functions (Wyman, Sloan, Shirley 2013 Gaussian fits), XYZ to
  linear sRGB, chromaticity-only lookup (brightness is applied as T^4 in
  the shader). Validated colors: 2000 K strongly red, 6500 K near white,
  20000 K strongly blue.
- redshift.py: covariant g = nu_obs / nu_em for prograde circular
  equatorial emitters in Kerr-Schild coordinates, float64 mirror of the
  shader code. Validated against the exact face-on Schwarzschild result
  g = sqrt(1 - 3M/r) to 0.2 percent through the full traced-ray chain.

Real-time disk rendering:
- The fragment shader detects equatorial plane crossings between
  consecutive RK4 positions, interpolates the crossing linearly, and
  terminates rays on the opaque disk (status 3). Emission combines the
  T(r) lookup texture (unit 0), the blackbody chromaticity texture
  (unit 1), the covariant redshift factor with the static-camera lapse,
  a T^4 brightness law, optional procedural streaks, and Reinhard tone
  mapping. Because a blackbody at T redshifts to a blackbody at g T,
  the pair (color lookup at g T, brightness (g T)^4) captures the exact
  g^4 bolometric scaling with no separate beaming factor.
- The engine caches the temperature lookup texture keyed by (spin,
  outer radius) and clamps the effective outer radius above the ISCO;
  the static-observer lapse is clamped to 1 inside the ergosphere where
  no static frame exists.
- Cross-validation: the float64 reference mirror gained the identical
  crossing detection; a GL test isolates the shader disk mask by
  differencing disk-on and disk-off frames and requires at least 0.98
  per-pixel agreement with the reference. A second GL test asserts warm
  blackbody colors and at least 1.5x Doppler beaming asymmetry.

Dynamics (blackhorizon/dynamics/):
- peters.py: Peters (1964) orbit-averaged da/dt and de/dt with the
  eccentricity enhancement factors, circular coalescence time, and an
  adaptive RK4 track integrator. Validated against the closed form
  a(t)^4 = a0^4 - (256/5) m1 m2 M t to 0.1 percent and the analytic
  merger time to 2 percent.
- pn_nbody.py: Newtonian plus 1PN Einstein-Infeld-Hoffmann n-body
  accelerations (Newhall, Standish, Williams 1983 form, Newtonian
  right-hand-side accelerations) and pairwise 2.5PN radiation reaction
  in the Iyer-Will form a = (8/5)(m1 m2 / r^3)[(v.n) n (3 v^2 +
  17 M / (3r)) - v (v^2 + 3 M / r)], distributed with mass-ratio
  factors that keep the Newtonian center of mass inertial. Validated:
  test-mass periapsis precession matches 6 pi M / (a (1 - e^2)) to
  2 percent; circular secular decay matches the integrated Peters
  closed form to 1 percent; eccentric e = 0.4 secular decay matches
  the Peters enhancement to 1 percent. Note: the EIH equations are
  Lorentz rather than Galilean invariant, so the Newtonian center of
  mass is only conserved with 1PN off; the corresponding test targets
  the radiation-reaction distribution.
- tde.py: tidal radius, Hills mass, normalized fallback rate with the
  t^(-5/3) full and t^(-9/4) partial disruption slopes, frozen-in
  energy spread estimate, and a debris stream generator that launches
  the star at the relativistically marginally bound speed (conserved
  E = 1 exactly, found by bisection inside the local light cone) and
  freezes particle orbits at pericenter into Stage 1 geodesic states.
  Validated: bound fraction 0.50 for parabolic encounters, energy
  spread consistent with M R / r_t^2, and debris Hamiltonians conserved
  at -1/2 through 400 RK4 steps.

### Stage 3 defaults

Disk enabled, outer radius 18 M, peak temperature 6500 K, exposure 1.0
(1.5 headless), detail 1.0. The headless renderer now defaults to the
starfield background.

### Deferred to Stage 4

Offline maximum-fidelity rendering (bilinear-refined disk crossings,
higher-order interpolation, FP64 promotion near the photon shell),
video export, disk turbulence evolved in time, plunging-region
emission, and debris self-gravity.


## Stage 4 addendum: offline rendering, video, symplectic integration (implemented)

### Offline maximum-fidelity renderer (blackhorizon/offline/render.py)

The authoritative image path, which the real-time GLSL renderer
approximates:
- Float64 adaptive Dormand-Prince 5(4) tracing (first pass rtol 1e-9),
  array-module generic so the gpu backend (CuPy) accelerates it; rays
  are traced in memory-bounded tiles.
- Disk crossings are localized by bisection: from the pre-step state, a
  fractional Runge-Kutta substep samples the same trajectory at fourth
  order, and forty bisections pin the equatorial crossing to a 1e-12
  fraction of the accepted step, replacing the real-time renderer's
  linear interpolation. Rays whose refined radius misses the annulus
  correctly continue through the inner gap or past the rim.
- Photon-shell fidelity ladder: rays that consume most of the step
  budget (they wind near the photon shell where trajectories are
  exponentially sensitive) are retraced with rtol 1e-11 and a
  quadrupled budget before classification. This realizes the design
  goal of promoted precision near the shell; the pipeline is float64
  end to end, so the promotion is in tolerance rather than word size.
- Subpixel supersampling on a regular grid (default 2x2 per pixel),
  averaged after shading.
- Emission is linear HDR: the Page-Thorne temperature profile and the
  covariant redshift factor as in Stage 3 but with a 2048-entry
  temperature table and a 1024-entry blackbody table interpolated in
  float64, plus a deterministic HDR starfield with hash-placed
  blackbody-tinted stars and a faint galactic band.
- Development happens in post (blackhorizon/offline/post.py): bloom as
  a thresholded bright pass under a separable Gaussian, the ACES filmic
  tone curve (Narkowicz 2015 fit), and exact sRGB encoding.

Validated by tests: shadow, disk, and HDR-exceeding highlights present
in a rendered frame; bit-exact determinism across runs; supersampling
strictly reduces edge gradient energy; post-processing properties
(bloom only brightens and ignores sub-threshold pixels, ACES is
monotonic and bounded, blur preserves the mean).

### Camera paths and video (blackhorizon/offline/camera_path.py, video.py)

Keyframed orbital camera paths (distance, inclination, azimuth, field
of view) with smoothstep easing; azimuth interpolates unwrapped so
multi-revolution orbits are single segments. Video rendering drives the
real-time GLSL engine headlessly along a path at a supersampling
multiple of the target resolution, box-downsamples, applies a light
per-frame bloom, and encodes H.264 through imageio-ffmpeg (the "video"
optional extra). A frames-to-mp4 utility encodes existing PNG
sequences, including the TDE demo output. The maximum-fidelity float64
path remains the tool for hero stills; the GLSL path keeps a full
orbit affordable.

### Symplectic integration (integrators.implicit_midpoint_step)

The implicit midpoint rule, solved by fixed-point iteration, is second
order and symplectic for the geodesic Hamiltonian flow. At large fixed
steps over tens of thousands of steps the RK4 Hamiltonian error grows
strictly monotonically (secular drift) while the midpoint error
oscillates and returns, staying bounded; both behaviors are asserted
by tests, along with second-order convergence. Use it for long orbital
evolutions such as debris streams; adaptive Dormand-Prince remains the
tool for imaging rays. A fully explicit Kerr-split symplectic scheme
(Wu et al. 2021) remains future work; the implicit midpoint rule
provides the structure preservation the design called for with a
simple, testable construction.

### TDE demo rendering

Particles now draw as additive Gaussian splats with a fading trail
buffer and a fixed field of view (default five tidal radii, CLI
--extent) so trails and encoded videos stay geometrically consistent;
frame brightness auto-normalizes to its 99.5th percentile under an
asinh stretch. Encode the frames with:
python -m blackhorizon.offline.video --encode-frames tde_frames
--fps 12 --output tde.mp4

### Performance notes

Container reference (CPU, float64): 160x100 at supersample 1 renders in
about 10 seconds; cost scales linearly in rays. The gpu backend runs
the same tracing loop on CuPy (measured in Stage 1 at roughly 5.7x the
CPU ray-step rate in float64 on the RTX 3070). Video via the GLSL
engine renders full frames in milliseconds on native hardware.


## Stage 4 field acceptance (user hardware, RTX 3070)

- Full test suite: 103 passed in 52 s.
- Offline hero frame, 1920 x 1200 at supersample 2 (9.2 million float64
  rays), gpu backend: 1669 s. The throughput dip between roughly 25 and
  50 percent of the rays is the tile band whose rays wind near the
  photon shell: they take the most adaptive steps and trigger the
  tolerance-tightening refinement pass, and the shrinking active set
  underutilizes the GPU until the band passes.
- Orbit video, 300 frames at 1280 x 720, ultra preset, supersample 2
  (2560 x 1440 renders downsampled): 150 s total, 2 frames per second
  including H.264 encoding.

## Stage 4 bug fix: disk orbit convention for negative spin

The disk co-rotates in +phi everywhere in the pipeline (its angular
velocity is Omega = 1 / (r^(3/2) + a)), but the inner edge was taken
from isco_radius(prograde=True), which means co-rotating with the hole.
For negative spin the +phi orbits are retrograde relative to the hole,
so the signed-a Page-Thorne bracket received the wrong x0 and clamped
to zero over a band of negative spins, crashing the real-time app when
the spin slider swept through them. The fix is the shared helper
emission.novikov_thorne.disk_inner_radius, which selects the ISCO
branch of the +phi orbits; the engine, the offline renderer, and the
profile functions all use it. Regression tests sweep the full slider
range for finite normalized tables and assert the physical efficiency
ordering: retrograde disks truncate and peak farther out (peak radius
14.1 M at a = -0.9 versus 9.55 M at a = 0 versus 3.44 M at a = +0.9).

## Stage 4 polish

- blackhorizon.offline imports its submodules lazily (PEP 562), so
  python -m blackhorizon.offline.render no longer trips runpy's
  double-import warning.
- The TDE demo deposits particle splats into the fading trail buffer
  many times per output frame (--deposits-per-frame, default 24, sized
  so the per-deposit displacement stays below the splat width), so
  fast-moving debris paints a continuous fading ribbon; frames are
  captured before advancing, so frame zero shows the star at
  pericenter. Reproduced field numbers on the RTX 3070: hero frame
  1645 to 1669 s across runs, orbit video 150 to 159 s for 300 frames.


## Stage 5 and beyond: rendering the black hole interior (research findings and plan)

### Corrected physics for the interior view

Research pass conducted against the primary literature; full citations
in the accompanying research report. Three user intuitions were checked:

1. "The window on the universe shrinks as we fall in, closing at the
   horizon." Partially corrected: the shrinking is relativistic
   aberration from the camera's inward velocity, not the horizon
   closing the sky. For a rain (Painleve-Gullstrand) infaller the
   outside universe still fills more than half the sky at horizon
   crossing (NASA Schnittman and Powell 2024 render this explicitly)
   and narrows into a bright horizontal band only near the singularity,
   where tidal aberration diverges (Hamilton and Polhemus,
   arXiv:0903.4717).
2. "Inside, every direction faces inward." Restated precisely: inside
   the horizon the radial coordinate becomes timelike; decreasing r is
   as inevitable as the advance of time, and the singularity is a
   moment in the observer's future, not a place in a direction. Thrust
   cannot increase r; it can only shorten the remaining proper time
   relative to the coasting geodesic (Lewis and Kwan 2007,
   arXiv:0705.1029).
3. "The Kerr singularity is a ring." Correct for the exact solution
   (x^2 + y^2 = a^2, z = 0 in our Kerr-Schild Cartesian coordinates),
   but the modern consensus is that the tidy interior of exact Kerr,
   including the inner Cauchy horizon at r_minus, the ring, closed
   timelike curves, and negative-r universes, does not survive in a
   real black hole. Mass inflation (Poisson and Israel 1989/1990)
   replaces the inner horizon with a weak null singularity (Ori 1991,
   proven at C0 level by Dafermos and Luk, arXiv:1710.01722) joined to
   a spacelike Marolf-Ori/BKL branch (Burko and Khanna,
   arXiv:1901.03413); quantum stress-energy independently diverges
   there (Zilberman et al., PRL 129, 261102).

### Why a "realistic interior" mode is implementable with current math

The Dafermos-Luk theorem is a stability statement: the perturbed
interior remains C0-close to exact Kerr all the way to the Cauchy
horizon, with the pathology confined to an exponentially thin layer at
r_minus where curvature blows up while the metric barely moves.
Rendering consequence: light propagation along essentially the whole
plunge is correctly computed on the exact Kerr geometry the project
already integrates. The realistic and idealized pictures diverge
visually only in the final approach to r_minus.

The dominant visible phenomenon there, the Penrose blue sheet, is
computable with existing machinery: outside-universe light piles up
along the inner horizon with blueshift growing as
exp(kappa_minus * v), kappa_minus = (r_plus - r_minus) /
(2 (r_minus^2 + a^2)). The covariant g-factor evaluated along traced
rays on exact Kerr yields this divergence quantitatively; rays are
terminated where the classical spacetime ends, with logarithmic tone
mapping of the saturating blueshift. The realistic mode is therefore a
quantitative rendering of the consensus picture, not a painted-on
wall. Full 3+1 numerical relativity of a generic rotating interior is
the only truly inaccessible rung and is unnecessary for visualization;
even research codes (Burko and Khanna) solve reduced problems.

### Planned stages

Stage 5, Schwarzschild-class interior plunge (doomed observer):
- Infalling camera tetrad (Doran/rain frame; Gram-Schmidt against the
  Kerr-Schild basis) replacing the static-observer ray construction;
  rays generated by local aberration from the tetrad. This also fixes
  exterior ergosphere rendering.
- Fully covariant redshift g = (p . u_cam) / (p . u_em) everywhere,
  replacing the static lapse.
- Ray reclassification: backward rays crossing the future horizon
  outward are the normal mechanism by which the infaller sees the sky
  (not capture); a new terminated-at-singularity class; explicit
  handling of the ingoing-Kerr-Schild coordinate barrier where backward
  rays asymptote to the outgoing horizon (rendered as the horizon
  image; see arXiv:2304.03804 on ingoing/outgoing form regularity).
- Doomed-observer camera: the camera is a proper-time-parameterized
  timelike geodesic (machinery shared with TDE debris). Presets: rain
  (E = 1), maximal-lifetime (tau_max = pi M from rest at the horizon;
  Toporensky and Zaslavskii, arXiv:1905.02150), fast plunge (E > 1,
  tau down to 4M/3 from infinity). Remaining-proper-time countdown
  HUD; thrusters re-aim or change worldline but never increase r and
  never extend life beyond the coasting geodesic; the reset button is
  the only exit. Render mode gains programmable infall trajectories on
  the camera-path system.
- Realtime UI additions: interior-mode toggle with pedagogical
  overlays (countdown, r readout, light-cone tilt glyphs, river-model
  flow vectors per Hamilton and Lisle 2008, optional geodesic bundle
  lines), starfield/checkerboard background toggle, reset camera
  button.
- Acceptance benchmark: reproduce Hamilton's Schwarzschild bubble
  (outgoing horizon ahead, ingoing behind) and the near-singularity
  horizontal blueshifted band.

Stage 6, Kerr interior with inner-horizon physics:
- Mode A (default, realistic): exact Kerr to near r_minus plus the
  quantitatively computed blue-sheet divergence and termination;
  late-infall spacelike branch terminates at small r.
- Mode B (labeled idealized analytic extension): continue through
  r_minus, the ring (with a toroidal proximity guard and regularized
  stepping), and optionally into the negative-r region through the
  disk x^2 + y^2 < a^2. Clearly labeled non-physical; float64 offline
  primary, realtime as stretch goal with cutoffs.

Stage 7 (optional, research-grade): dynamical mass-inflation
fly-through on the charged-Vaidya/Ori semi-analytic surrogate
(Hamilton's approach: charge as surrogate for spin). Requires
promoting the Hamiltonian tracer to time-dependent metrics (p_t no
longer conserved; RHS gains a dH/dt term); float64 offline only.

### Architecture and stack impact

No new dependencies. Changes concentrate in: geodesics (tetrad
construction, non-autonomous Hamiltonian for Stage 7), tracer/render
(ray classes, barrier handling, blueshift saturation), realtime
(camera-state machine for infall mode, overlay shaders, UI toggles),
offline (doomed-observer camera paths). Precision policy: interior is
float64-offline primary; float32 realtime interior for Schwarzschild
with an inner cutoff is feasible, Kerr interior realtime is a stretch
goal. Hardware is not the constraint; the exponential blueshift
saturates float64 (ln g ~ 700) long before it stresses the GPU, and is
tone-mapped logarithmically.

## Stage 5 implementation summary (delivered)

Stage 5 is implemented and validated. The doomed-observer machinery:

- blackhorizon/frames.py: metric inner products, index raising and
  lowering for the Kerr-Schild metric, the rain (Doran) 4-velocity
  p = (-1, w l) with w = -sqrt(2H)/(1+sqrt(2H)) regular through both
  horizons, metric Gram-Schmidt tetrads, and tetrad camera rays with
  unit camera frequency by construction (so the covariant redshift is
  g = 1 / (p_traced . u_emitter) with no observer lapse).
- Per-ray conserved energy p_t threads through the GLSL shader, the
  float64 reference, the offline renderer, and the redshift module;
  the exterior p_t = +1 path is unchanged and revalidated.
- Interior ray rules: rays from inside cross the horizon freely
  (backward in time) and terminate near the singularity
  (RayStatus.TERMINATED); the step heuristic is piecewise, keeping the
  validated (r - r_plus) scaling outside and r/2 inside, where no
  photon orbits exist and Kerr-Schild is regular at the crossing.
- realtime/infall.py: the camera worldline as a proper-time timelike
  geodesic with radius-adaptive substeps (proportional to r^1.5 near
  the center, tracking the sqrt(2M/r) plunge speed), exact
  rapidity-boost thrust in the local tetrad frame, and a lookahead
  countdown of the coasting proper time to the terminal radius.
- The app state machine: crossing the horizon hands the camera to the
  infall state; thrust stays available (physics itself forbids
  increasing r inside, no clamps needed; burns generally shorten life,
  and braking toward the E = 0 trajectory lengthens it, verified
  against Lewis and Kwan 2007); the reset button is the only exit.
  The panel shows a proper-time countdown, an interior banner, a
  time-scale slider, starfield and overlay toggles, and a river-model
  light-cone glyph (Hamilton and Lisle 2008).
- offline/infall.py: programmable plunge rendering with rain, maximal
  (E = 0), and fast (E > 1) presets, per-frame infalling tetrads, and
  the maximum-fidelity float64 pipeline; the starfield shifts by g^4
  in intensity with a first-order chromatic slide of the star palette.

Validated physics anchors (tests/test_interior.py and friends):

- Exact factor-2 redshift of overhead starlight for the rain observer
  at the horizon: g = 1 / (1 + sqrt(2M/r)).
- Maximal interior lifetime: the E = 0 geodesic reproduces the
  analytic tau integral to 0.1 percent; the full-range value is pi M.
- More than half the sky shows the outside universe just inside the
  horizon, narrowing with depth but never closing (NASA Schnittman
  and Powell 2024; Hamilton and Polhemus arXiv:0903.4717).
- The radius never increases inside for any future-directed timelike
  worldline, including maximal outward-momentum launches.
- Rain interior transit 4M/3; rain from r0 gives
  tau = (2/3) r0^(3/2) / sqrt(2M), matched by the worldline builder.
- Disk light seen transversely from deep inside is blueshifted
  (g up to ~6.5 at r = 1.2M), rendering blue-white; radially outward
  sky light is redshifted and dims as g^4. Both match Hamilton's
  near-singularity analysis and are exercised by GL and offline tests.

Suite: 115 tests passing (105 exterior from Stages 1-4, 10 interior).

### Stage 5 field fixes: Kerr interior termination and GPU stability

Field testing on an RTX 3070 (spin 0.9, the app default) surfaced two
defects with one root cause: Stage 5's Schwarzschild-class plunge
logic let both the camera worldline and the rays pass the inner
(Cauchy) horizon. Between the horizons r is timelike and strictly
decreasing, but inside r_minus it turns spacelike again: the worldline
left the mass shell under the stiff near-ring fields (Hamiltonian
drifting from -1/2 to +76 within two frames), the camera 4-velocity
went non-timelike, and tetrad construction raised. Separately, at the
ultra preset (2048 steps, full resolution) rays penetrating past
r_minus blueshift exponentially in the blue-sheet region, the
momentum-clamped steps collapse, and a single draw call exceeds the
GPU driver watchdog: the reported hard system hang.

The fix is the physically correct one and anticipates Stage 6: for
spinning holes the terminal surface is the Cauchy horizon. Realistic
infall ends at the infinitely blueshifted blue sheet there
(Poisson-Israel mass inflation), and Dafermos-Luk (arXiv:1710.01722)
guarantee the geometry down to that surface matches exact Kerr, so
everything Stage 5 renders remains exact. Idealized continuation past
r_minus stays Stage 6 material behind an explicit label. Concretely:

- InfallState clamps its stop radius to max(stop, 1.02 r_minus), and
  monotonicity of r between the horizons guarantees termination.
- The worldline renormalizes its momentum to the exact mass shell
  each frame (drift-proof against stiff fields); an irrecoverable
  state restores the last well-conditioned one and terminates there,
  never handing the renderer a degenerate position.
- The engine sends rays the same terminal surface (u_interior_stop is
  clamped to 1.02 r_minus) so no ray integrates the blue-sheet
  region, wraps tetrad construction with a rain-observer fallback,
  and clamps degenerate camera positions onto the terminal sphere.
- Interior rendering caps its step budget at 1024 (interior views
  have no photon-shell winding needing ultra's 2048), bounding the
  worst-case draw call below the watchdog; the reference mirrors the
  cap and the interior step floor exactly.

Regression coverage: Kerr crossing with per-frame tetrad construction
through termination and beyond, thrust spam through the full plunge,
a GL render with a deliberately corrupted camera 4-velocity, and
Schwarzschild stop-radius invariance. Suite: 119 tests passing.

### Journey modes: realistic and idealized (user selectable)

Following the field decision, both stances on the Cauchy horizon are
now user options, with an explicit statement of the simulation's
standing assumptions: the observer is an indestructible test particle
(the same grant that lets them survive the radiation environment and
tidal forces outside), and whether anything exists beyond the Cauchy
horizon is not testable today; current astrophysics says it does not
survive in real holes, and the simulator defaults accordingly.

- realistic (default): the journey of a spinning hole terminates at
  the Cauchy horizon, where the blue sheet ends any physical infall.
  Everything rendered is exact Kerr (Dafermos-Luk continuity).
- idealized: continue into the inner region of exact eternal vacuum
  Kerr, clearly labeled non-physical. The single stationary ingoing
  Kerr-Schild chart covers the inward crossing of both horizons, and
  interior cameras there see the outside universe through the Cauchy
  horizon (about 89 percent of the sky at r = 0.4 for a = 0.9,
  reference validated). Radius may legally increase inside r_minus.
  Two worldline classes leave the chart and terminate with reason
  "chart": trajectories reaching the ring plane (the gateway to the
  negative-r antiverse, which needs the second sheet of the radius
  quartic and remains future work), and outgoing branches that
  asymptote the Cauchy horizon attempting the maximal-extension exit
  into another universe. The mass shell is held exactly by per-frame
  renormalization; the mode is switchable mid-flight from the panel
  (--journey in the offline CLI).

The offline Doran-family builder was corrected in the process: the
E-parameterized radial infall momentum p = (-E, w l) must align with
the Kerr-Schild null vector l, which coincides with the radial
direction only at zero spin; the mass-shell verification caught the
inconsistency at a = 0.9.

Suite: 124 tests passing, including rain across the Cauchy horizon
with the Hamiltonian held to 1e-6, the legal radius increase inside
r_minus, chart-boundary classification, live journey switching, and
GL rendering from inside the Cauchy horizon in both modes.

## Stage 6 implementation summary (delivered)

Stage 6 makes the realistic Kerr interior quantitative: the blue
sheet is now computed and rendered, not just a terminal surface.

Physics model (blackhorizon/emission/bluesheet.py). A real hole is
illuminated at all advanced times v (starlight, CMB, its own disk);
radiation entering at v reaches an observer near the Cauchy horizon
amplified by exp(kappa_minus v) with kappa_minus = (r_plus - r_minus)
/ (2 (r_minus^2 + a^2)) (Poisson-Israel 1990; Ori 1991;
Hamilton-Avelino arXiv:0811.1926). Along infalling worldlines the
near-horizon relation v = -(1/kappa_minus) ln x, with proximity
x = (r - r_minus)/(r_plus - r_minus), closes the law: every external
ray is amplified by B = x_match / x below the matching proximity
(continuous, capped at 60 for the physics readout). Dafermos-Luk
continuity licenses the split: exact vacuum Kerr supplies the
geometry, aberration, and covariant per-ray shifts everywhere along
the approach; the blue sheet enters purely as B multiplying the
observer lapse, so the disk and starfield flare through the same
covariant pipeline with no new geometric machinery. Rays terminating
on the inner-horizon surface render the sheet itself with radiance
following the B^4 law from a faint base. Idealized journey mode
(eternal vacuum, no radiation) and Schwarzschild (no inner horizon)
get identity amplification, keeping both honest.

Display adaptation. The physical amplification diverges; a display,
like any eye, adapts. The intensity gain caps at B_display = 8
(4096-fold), preserving per-ray structure through the approach, and
a smoothstep whiteout ramp over true B in [8, 30] carries the
divergence to the terminal white bath; the HUD reports the true
uncapped value. The experiential sequence, validated in GL: dark
approach, blue glow from about x = 0.35, an intense blue flare near
x = 0.1 (view mean up 60-fold, blue-dominant), blinding at the
display cap, whiteout at the wall. Offline applies the identical law
(observer lapse, starfield shift including a chromatic slide of the
galactic band, sheet radiance, HDR whiteout mix), so realtime and
render modes agree.

HUD: a cyan blue-sheet amplification readout ramps during the
realistic Kerr approach, and the terminal message names the blue
sheet. All Stage 5 hardening carries over unchanged: Cauchy-horizon
terminal surfaces, mass-shell renormalization, chart-boundary
classification, the interior step floor and budget cap, and the
engine tetrad fallback.

Suite: 130 tests passing, adding the kappa_minus analytic check,
amplification monotonicity and matching continuity, zero-spin and
idealized identity, display-law properties, offline flare-brightening
with idealized darkness at the same position, and the GL flare
progression.

## Stage 7 implementation summary (delivered, offline only)

Stage 7 replaces the Stage 6 whiteout with a rendered flythrough of
the mass-inflation layer, using the spherical charged-Vaidya/Ori
surrogate that makes tractable what Kerr's angular structure does not.

The surrogate (blackhorizon/vaidya.py). An ingoing charged-Vaidya
spacetime in the same Cartesian Kerr-Schild form as the rest of the
package (g = eta + 2 H l l, H = m(v, r)/r - q^2/(2 r^2), v = t + r),
so every tetrad and imaging tool applies unchanged. The charge q maps
to Kerr's spin a by matching the horizon structure r_pm = M pm
sqrt(M^2 - q^2); q = 0.9 reproduces the a = 0.9 horizons to 1e-5. A
Price power-law tail m1(v) = M - dm (v_tail/v)^(p-1), p = 12, feeds
the influx; an outgoing Ori null shell is integrated exactly from
dR/dv = f1/2, and the Dray-'t Hooft-Redmount matching dm2/dv =
L(v) f2/f1 is co-integrated along it for the inflating region-II mass
m2(v). Because the metric is time dependent, photon energy is not
conserved: vaidya_geodesic_rhs supplies the extended Hamiltonian flow
with dp_t/dlambda = (dH/dt)(l . p)^2, verified against central
differences to 1e-6, evolving p_t under the tail and reducing exactly
to the conserved stationary flow when the fluxes vanish.

Analytic anchors (tests/test_vaidya.py): the inner surface gravity
kappa_minus = (r_plus - r_minus)/(2 r_minus^2) to 1e-12; the mass
function jumps 1e4-fold across the shell in the strong-inflation
regime; and the inflation e-folding rate equals kappa_minus to within
5 percent, the classic Poisson-Israel-Ori result, once the v^(-p)
Price prefactor is included in the fit (a naive pure-exponential fit
misreads the polynomial modulation as a 20 percent error).

The flythrough renderer (blackhorizon/offline/inflation.py). The
camera rides the surrogate's ingoing rain worldline from between the
horizons, through the shell, to the render horizon at the inner
horizon; frames are sampled evenly in proper time, each building a
frozen-slice tetrad and tracing past-directed rays with the full
time-dependent flow. Escaped rays shade to a surrogate starfield;
rays ending on the inflating layer glow blue-white by the local
Misner-Sharp mass. The Stage 5 tetrad-fallback guard is carried over,
so a diverged worldline sample can never crash a frame.

Physics scope, stated honestly. A radial geodesic camera crosses the
Cauchy horizon at finite, mild Misner-Sharp mass: the exponential
inflation lives at late advanced time along the horizon, which a fast
infaller outruns (strong inflation is an observer-who-lingers
phenomenon). Stage 6 already renders what the infaller sees diverging
(the blue-sheet flare); Stage 7 renders the geometry of the layer the
infaller passes through. Both are correct and complementary. Rendering
the violent late-CH region would require a non-geodesic (hovering)
camera or a shell-comoving parameterization, a possible future
refinement; the surrogate already contains that geometry (the mass
function diverges along the shell), only the camera samples the mild
early portion.

Suite: 140 tests passing, adding horizon-structure and surface-gravity
checks, the Hamiltonian-flow and static-limit and time-dependence
tests, the mass-inflation e-folding and divergence and cross-shell
jump tests, and a flythrough worldline-plus-frame integration test.
This closes the planned Stage 1 through 7 arc: exterior disk and
photon ring, the doomed plunge, the Schwarzschild interior, the Kerr
interior with selectable journey modes, the quantitative blue sheet,
and now a rendered passage through the mass-inflation layer.
