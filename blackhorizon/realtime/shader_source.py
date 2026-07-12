"""Shader source assembly.

Loads GLSL sources bundled with the package and injects compile-time
defines directly after the #version line, keeping the shader files
themselves valid GLSL for editors and linters.
"""

from __future__ import annotations

from importlib import resources

_SHADER_PACKAGE = "blackhorizon.realtime.shaders"


def load_shader(filename: str) -> str:
    """Read a shader source file bundled with the package."""
    return (
        resources.files(_SHADER_PACKAGE).joinpath(filename).read_text()
    )


def inject_defines(source: str, defines: dict[str, object]) -> str:
    """Insert #define lines immediately after the #version directive."""
    lines = source.splitlines()
    if not lines or not lines[0].startswith("#version"):
        raise ValueError("shader source must start with a #version line")
    define_lines = [
        f"#define {name} {value}" for name, value in defines.items()
    ]
    return "\n".join([lines[0], *define_lines, *lines[1:]]) + "\n"


def vertex_source() -> str:
    """Assembled fullscreen vertex shader source."""
    return load_shader("fullscreen.vert")


def fragment_source(max_steps_limit: int) -> str:
    """Assembled Kerr tracer fragment shader source.

    Args:
        max_steps_limit: Compile-time loop bound; the runtime uniform
            u_max_steps must stay at or below this.
    """
    return inject_defines(
        load_shader("kerr_tracer.frag"), {"MAX_STEPS": int(max_steps_limit)}
    )
