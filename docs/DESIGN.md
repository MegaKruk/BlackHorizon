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
