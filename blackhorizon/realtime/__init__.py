"""Real-time interactive rendering of Kerr black holes (Stage 2).

The lensed view is traced entirely inside a GLSL fragment shader that
transcribes the validated Stage 1 physics; see docs/DESIGN.md sections
2.2 to 2.4 and the Stage 2 addendum. The windowed viewer lives in
app; single frames can be rendered without a window via headless.
"""

from .engine import KerrRenderEngine
from .fly_camera import FlyCamera
from .settings import BackgroundMode, QualityPreset, RenderSettings

__all__ = [
    "KerrRenderEngine",
    "FlyCamera",
    "RenderSettings",
    "QualityPreset",
    "BackgroundMode",
]
