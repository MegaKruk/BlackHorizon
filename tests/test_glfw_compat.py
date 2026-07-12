"""Tests for the pyGLFW / imgui-bundle GLFW alignment shim."""

import os

import pytest

from blackhorizon.realtime.glfw_compat import (
    ensure_compatible_glfw,
    find_imgui_bundle_glfw,
)

imgui_bundle_installed = find_imgui_bundle_glfw() is not None


class TestGlfwCompat:
    def test_respects_user_override(self, monkeypatch):
        monkeypatch.setenv("PYGLFW_LIBRARY", "/custom/libglfw.so")
        ensure_compatible_glfw()
        assert os.environ["PYGLFW_LIBRARY"] == "/custom/libglfw.so"

    @pytest.mark.skipif(
        not imgui_bundle_installed,
        reason="imgui-bundle with bundled GLFW not installed",
    )
    def test_points_pyglfw_at_bundled_library(self, monkeypatch):
        monkeypatch.delenv("PYGLFW_LIBRARY", raising=False)
        ensure_compatible_glfw()
        library = os.environ.get("PYGLFW_LIBRARY")
        assert library is not None
        assert library.endswith("libglfw.so.3")
        assert os.path.exists(library)

    def test_noop_without_imgui_bundle(self, monkeypatch):
        monkeypatch.delenv("PYGLFW_LIBRARY", raising=False)
        monkeypatch.setattr(
            "blackhorizon.realtime.glfw_compat.find_imgui_bundle_glfw",
            lambda: None,
        )
        ensure_compatible_glfw()
        assert "PYGLFW_LIBRARY" not in os.environ
