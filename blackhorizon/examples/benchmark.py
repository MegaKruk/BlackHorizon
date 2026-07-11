"""Benchmark the geodesic integration throughput.

Usage:
    python -m blackhorizon.examples.benchmark [--rays N] [--steps K]

Reports nanoseconds per ray-step for fixed-step RK4 (raw right-hand-side
throughput) on the CPU backend and, when available, on the GPU backend.
Compare against the design target in docs/DESIGN.md section 4.
"""

from __future__ import annotations

import argparse
import time

from ..backend import get_xp, gpu_available
from ..geodesics import build_state, geodesic_rhs, null_momentum_from_velocity
from ..integrators import rk4_step
from ..kerr import KerrSpacetime


def make_states(xp, n_rays: int, dtype) -> tuple:
    """Build a batch of inward-pointing photons on a distant shell."""
    spacetime = KerrSpacetime(mass=1.0, spin=0.9)
    rng_host = __import__("numpy").random.default_rng(7)
    u = rng_host.normal(size=(n_rays, 3))
    u /= __import__("numpy").linalg.norm(u, axis=-1, keepdims=True)
    positions = xp.asarray(50.0 * u, dtype=dtype)
    directions = xp.asarray(-u, dtype=dtype)
    momenta = null_momentum_from_velocity(spacetime, positions, directions)
    return spacetime, build_state(positions, momenta)


def run_backend(backend: str, n_rays: int, n_steps: int, dtype_name: str) -> None:
    """Time fixed-step RK4 on one backend and print the throughput."""
    xp = get_xp(backend)
    dtype = getattr(xp, dtype_name)
    spacetime, state = make_states(xp, n_rays, dtype)
    h = xp.full((n_rays,), 0.05, dtype=dtype)

    def rhs(batch):
        return geodesic_rhs(spacetime, batch)

    # Warm up (JIT-free, but first calls allocate and, on GPU, compile).
    state = rk4_step(rhs, state, h)
    if backend == "gpu":
        xp.cuda.runtime.deviceSynchronize()

    start = time.perf_counter()
    for _ in range(n_steps):
        state = rk4_step(rhs, state, h)
    if backend == "gpu":
        xp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - start

    ray_steps = n_rays * n_steps
    ns_per = elapsed / ray_steps * 1e9
    print(
        f"{backend} ({dtype_name}): {n_rays} rays x {n_steps} RK4 steps "
        f"in {elapsed:.2f} s = {ns_per:.1f} ns per ray-step "
        f"({ray_steps / elapsed / 1e6:.2f} M ray-steps/s)"
    )


def main() -> None:
    """Run the benchmark on all available backends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    run_backend("cpu", args.rays, args.steps, "float64")
    if gpu_available():
        run_backend("gpu", args.rays, args.steps, "float64")
        run_backend("gpu", args.rays, args.steps, "float32")
    else:
        print("gpu: not available (install cupy-cuda12x and NVIDIA driver)")


if __name__ == "__main__":
    main()
