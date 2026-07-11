"""Black Horizon: an interactive black hole and orbital dynamics simulator.

Stage 1 provides the core general-relativistic engine: Kerr spacetime in
horizon-penetrating Kerr-Schild coordinates, Hamiltonian geodesic
integration with adaptive error control on CPU (NumPy) or GPU (CuPy), and
a validated shadow ray tracer. See docs/DESIGN.md for the full design.
"""

from .backend import gpu_available
from .camera import PinholeCamera
from .kerr import KerrSpacetime
from .tracer import RayStatus, TraceResult, trace_rays

__version__ = "0.1.0"

__all__ = [
    "KerrSpacetime",
    "PinholeCamera",
    "RayStatus",
    "TraceResult",
    "trace_rays",
    "gpu_available",
    "__version__",
]
