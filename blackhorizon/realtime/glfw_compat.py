"""Compatibility shim aligning pyGLFW with imgui-bundle's GLFW library.

The problem this solves: imgui-bundle's native module is dynamically
linked against its own bundled libglfw.so.3, whose build requires X11
symbols (for example glfwGetX11Window). The pyGLFW package, meanwhile,
selects between bundled X11 and Wayland-only builds of GLFW based on the
session type. On a Wayland desktop pyGLFW loads its Wayland-only build
first; because both libraries carry the SONAME libglfw.so.3, the dynamic
linker then reuses that already-loaded library for imgui-bundle, whose
import fails with an undefined X11 symbol.

The fix: before pyGLFW loads anything, point it at imgui-bundle's own
GLFW library via the PYGLFW_LIBRARY environment variable, so the whole
process shares a single GLFW. On Wayland desktops the window then runs
through XWayland, which GLFW selects automatically.

ensure_compatible_glfw must be called before the first import of glfw.
"""

from __future__ import annotations

import os
from importlib import util
from pathlib import Path


def find_imgui_bundle_glfw() -> Path | None:
    """Path of the GLFW library bundled with imgui-bundle, if installed."""
    spec = util.find_spec("imgui_bundle")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "libglfw.so.3"
        if candidate.exists():
            return candidate
    return None


def ensure_compatible_glfw() -> None:
    """Make pyGLFW and imgui-bundle share one GLFW library.

    Respects a user-provided PYGLFW_LIBRARY. Does nothing when
    imgui-bundle is not installed or ships no GLFW library, in which case
    pyGLFW's own selection is fine because nothing else links GLFW.
    """
    if "glfw" in globals() or "PYGLFW_LIBRARY" in os.environ:
        return
    library = find_imgui_bundle_glfw()
    if library is not None:
        os.environ["PYGLFW_LIBRARY"] = str(library)
