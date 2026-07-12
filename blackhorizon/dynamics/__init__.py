"""Dynamics beyond single-particle geodesics: PN N-body, inspiral, TDEs."""

from .peters import coalescence_time_circular, integrate_inspiral
from .pn_nbody import NBodySystem, step_rk4, total_acceleration
from .tde import fallback_rate, generate_debris_stream, tidal_radius

__all__ = [
    "coalescence_time_circular",
    "integrate_inspiral",
    "NBodySystem",
    "step_rk4",
    "total_acceleration",
    "fallback_rate",
    "generate_debris_stream",
    "tidal_radius",
]
