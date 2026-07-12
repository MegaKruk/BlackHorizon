"""Fly camera tests: basis orthonormality, rotation, movement."""

import math

import numpy

from blackhorizon.realtime.fly_camera import FlyCamera


class TestBasis:
    def test_orthonormal_for_many_orientations(self):
        camera = FlyCamera()
        rng = numpy.random.default_rng(5)
        for _ in range(200):
            camera.yaw = rng.uniform(-math.pi, math.pi)
            camera.pitch = rng.uniform(-1.5, 1.5)
            forward, right, up = camera.basis()
            for v in (forward, right, up):
                assert abs(numpy.linalg.norm(v) - 1.0) < 1e-12
            assert abs(numpy.dot(forward, right)) < 1e-12
            assert abs(numpy.dot(forward, up)) < 1e-12
            assert abs(numpy.dot(right, up)) < 1e-12

    def test_right_handedness(self):
        camera = FlyCamera(yaw=0.0, pitch=0.0)
        forward, right, up = camera.basis()
        numpy.testing.assert_allclose(forward, [1.0, 0.0, 0.0], atol=1e-12)
        numpy.testing.assert_allclose(right, [0.0, -1.0, 0.0], atol=1e-12)
        numpy.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-12)


class TestLookAt:
    def test_from_orbit_faces_origin(self):
        camera = FlyCamera.from_orbit(30.0, 85.0)
        forward, _, _ = camera.basis()
        to_origin = -camera.position / numpy.linalg.norm(camera.position)
        assert numpy.dot(forward, to_origin) > 0.9999

    def test_look_at_ignores_coincident_target(self):
        camera = FlyCamera(yaw=1.0, pitch=0.5)
        camera.look_at(camera.position)
        assert camera.yaw == 1.0
        assert camera.pitch == 0.5


class TestRotation:
    def test_pitch_clamped(self):
        camera = FlyCamera()
        camera.rotate(0.0, -1e6)
        assert camera.pitch <= math.radians(89.0) + 1e-12
        camera.rotate(0.0, 1e6)
        assert camera.pitch >= -math.radians(89.0) - 1e-12

    def test_positive_dx_looks_right(self):
        camera = FlyCamera(yaw=0.0, pitch=0.0)
        camera.rotate(100.0, 0.0)
        # Looking right means yaw decreases toward -y.
        assert camera.yaw < 0.0


class TestMovement:
    def test_forward_moves_along_view(self):
        camera = FlyCamera(yaw=0.0, pitch=0.0, move_speed=2.0)
        start = camera.position.copy()
        camera.move(numpy.array([1.0, 0.0, 0.0]), dt=0.5)
        numpy.testing.assert_allclose(
            camera.position - start, [1.0, 0.0, 0.0], atol=1e-12
        )

    def test_diagonal_not_faster(self):
        camera = FlyCamera(move_speed=1.0)
        start = camera.position.copy()
        camera.move(numpy.array([1.0, 1.0, 0.0]), dt=1.0)
        assert abs(numpy.linalg.norm(camera.position - start) - 1.0) < 1e-12

    def test_zero_input_is_noop(self):
        camera = FlyCamera()
        start = camera.position.copy()
        camera.move(numpy.zeros(3), dt=1.0)
        numpy.testing.assert_array_equal(camera.position, start)
