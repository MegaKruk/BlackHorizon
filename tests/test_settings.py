"""Settings, presets, and shader source assembly tests."""

import pytest

from blackhorizon.realtime.settings import (
    MAX_STEPS_HARD_LIMIT,
    BackgroundMode,
    QualityPreset,
    RenderSettings,
)
from blackhorizon.realtime.shader_source import (
    fragment_source,
    inject_defines,
    vertex_source,
)


class TestSettings:
    def test_defaults_are_valid(self):
        RenderSettings().validate()

    def test_presets_are_valid_and_ordered(self):
        base = RenderSettings()
        steps = []
        for preset in QualityPreset:
            configured = base.apply_preset(preset)
            configured.validate()
            steps.append(configured.max_steps)
        assert steps == sorted(steps), "presets must increase step budget"

    def test_preset_does_not_mutate_original(self):
        base = RenderSettings(max_steps=333)
        base.apply_preset(QualityPreset.ULTRA)
        assert base.max_steps == 333

    def test_validation_rejects_bad_values(self):
        for bad in (
            {"spin": 1.5},
            {"fov_degrees": 0.0},
            {"max_steps": 0},
            {"max_steps": MAX_STEPS_HARD_LIMIT + 1},
            {"step_scale": -0.1},
            {"min_step": 0.0},
            {"resolution_scale": 0.0},
            {"momentum_bailout": 0.5},
        ):
            with pytest.raises(ValueError):
                RenderSettings(**bad).validate()

    def test_escape_radius_derivation(self):
        settings = RenderSettings()
        assert settings.effective_escape_radius(30.0) == 100.0
        assert settings.effective_escape_radius(80.0) == 160.0
        explicit = RenderSettings(escape_radius=500.0)
        assert explicit.effective_escape_radius(30.0) == 500.0


class TestShaderSource:
    def test_sources_load_and_are_ascii(self):
        for source in (vertex_source(), fragment_source(1024)):
            assert source.startswith("#version 330 core")
            source.encode("ascii")

    def test_defines_injected_after_version(self):
        source = fragment_source(2048)
        lines = source.splitlines()
        assert lines[0].startswith("#version")
        assert lines[1] == "#define MAX_STEPS 2048"

    def test_inject_requires_version_line(self):
        with pytest.raises(ValueError):
            inject_defines("void main() {}", {"X": 1})

    def test_fragment_uses_expected_uniforms(self):
        source = fragment_source(1024)
        for uniform in (
            "u_cam_position",
            "u_spin",
            "u_max_steps",
            "u_step_scale",
            "u_momentum_bailout",
            "u_background_mode",
        ):
            assert uniform in source

    def test_background_modes_match_shader_convention(self):
        assert int(BackgroundMode.CHECKERBOARD) == 0
        assert int(BackgroundMode.STARFIELD) == 1
