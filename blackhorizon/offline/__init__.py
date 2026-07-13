"""Offline rendering: maximum-fidelity frames, camera paths, video.

Submodules are imported lazily (PEP 562) so running them with
python -m does not re-import the package contents and trip runpy's
double-import warning.
"""

_EXPORTS = {
    "CameraKeyframe": "camera_path",
    "CameraPath": "camera_path",
    "orbit_path": "camera_path",
    "develop": "post",
    "OfflineSettings": "render",
    "render_hdr": "render",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve exported names from their submodules on first access."""
    if name in _EXPORTS:
        from importlib import import_module

        module = import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
