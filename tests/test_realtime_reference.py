"""Accuracy validation of the real-time shader algorithm.

The fragment shader trades the Stage 1 adaptive error control for a fixed
step heuristic. These tests quantify that trade by comparing the exact
NumPy mirror of the shader algorithm against the validated adaptive
tracer, so any change to the heuristic or its presets that degrades
physical accuracy fails loudly.
"""

import numpy

from blackhorizon.geodesics import (
    build_state,
    conserved_quantities,
    null_momentum_from_velocity,
)
from blackhorizon.kerr import KerrSpacetime
from blackhorizon.realtime.reference import trace_like_shader
from blackhorizon.realtime.settings import QualityPreset, RenderSettings
from blackhorizon.tracer import RayStatus, trace_rays

DISTANCE = 30.0


def equatorial_fan(spacetime, offsets):
    """Initial states and impact parameters for an equatorial ray fan."""
    n = offsets.shape[0]
    positions = numpy.tile([[DISTANCE, 0.0, 0.0]], (n, 1))
    directions = numpy.stack(
        [-numpy.ones(n), offsets, numpy.zeros(n)], axis=-1
    )
    directions /= numpy.linalg.norm(directions, axis=-1, keepdims=True)
    momenta = null_momentum_from_velocity(
        spacetime, positions, directions, time_orientation="past"
    )
    state0 = build_state(positions, momenta)
    energy, l_z = conserved_quantities(state0)
    return state0, l_z / energy


def shaded_black(status):
    """Pixels the shader would color black: captured plus step-budget."""
    return (status == int(RayStatus.CAPTURED)) | (
        status == int(RayStatus.MAX_STEPS)
    )


def boundary_of(status, impact):
    """Capture boundary |b| as the midpoint across the transition."""
    black = shaded_black(status)
    magnitudes = numpy.abs(impact)
    return 0.5 * (magnitudes[black].max() + magnitudes[~black].min())


class TestShaderAlgorithmAccuracy:
    def test_all_presets_match_adaptive_tracer_for_kerr(self):
        """Every preset classifies an a = 0.9 fan exactly like Stage 1."""
        st = KerrSpacetime(spin=0.9)
        offsets = numpy.linspace(0.03, 0.40, 300)
        offsets = numpy.concatenate([offsets, -offsets])
        state0, _ = equatorial_fan(st, offsets)
        truth = trace_rays(
            st, state0, escape_radius=2.0 * DISTANCE, rtol=1e-10
        )
        for preset in QualityPreset:
            settings = RenderSettings(spin=st.spin).apply_preset(preset)
            ref = trace_like_shader(
                st, state0, settings, escape_radius=2.0 * DISTANCE
            )
            agreement = numpy.mean(
                shaded_black(ref.status) == shaded_black(truth.status)
            )
            assert agreement == 1.0, f"preset {preset.value} disagrees"

    def test_schwarzschild_boundary_at_medium_preset(self):
        """Medium preset reproduces b_c = 3 sqrt(3) M within 0.5 percent."""
        st = KerrSpacetime(spin=0.0)
        offsets = numpy.linspace(0.10, 0.25, 600)
        state0, impact = equatorial_fan(st, offsets)
        settings = RenderSettings(spin=0.0).apply_preset(
            QualityPreset.MEDIUM
        )
        ref = trace_like_shader(
            st, state0, settings, escape_radius=2.0 * DISTANCE
        )
        b_measured = boundary_of(ref.status, impact)
        b_exact = 3.0 * numpy.sqrt(3.0)
        assert abs(b_measured - b_exact) / b_exact < 5e-3

    def test_near_extremal_high_preset(self):
        """High preset stays exact even at a = 0.998 on both sides."""
        st = KerrSpacetime(spin=0.998)
        offsets = numpy.linspace(0.03, 0.40, 300)
        offsets = numpy.concatenate([offsets, -offsets])
        state0, _ = equatorial_fan(st, offsets)
        truth = trace_rays(
            st, state0, escape_radius=2.0 * DISTANCE, rtol=1e-10
        )
        settings = RenderSettings(spin=st.spin).apply_preset(
            QualityPreset.HIGH
        )
        ref = trace_like_shader(
            st, state0, settings, escape_radius=2.0 * DISTANCE
        )
        agreement = numpy.mean(
            shaded_black(ref.status) == shaded_black(truth.status)
        )
        assert agreement == 1.0

    def test_no_rays_lost_to_numerical_failure(self):
        """Deep plungers terminate as captured, never as NaN escapes."""
        st = KerrSpacetime(spin=0.998)
        offsets = numpy.linspace(0.001, 0.15, 200)
        state0, _ = equatorial_fan(st, offsets)
        settings = RenderSettings(spin=st.spin).apply_preset(
            QualityPreset.LOW
        )
        ref = trace_like_shader(
            st, state0, settings, escape_radius=2.0 * DISTANCE
        )
        assert bool(numpy.all(numpy.isfinite(ref.positions)))
        assert bool(
            numpy.all(ref.status == int(RayStatus.CAPTURED))
        ), "all rays in this fan plunge in the exact solution"
