"""Video rendering and encoding.

Renders camera-path animations with the real-time GLSL engine on a
headless offscreen context, which keeps a full orbit affordable, and
encodes H.264 with imageio-ffmpeg (install the "video" extra).
Fidelity levers on top of the engine: frames are rendered at a
supersampling multiple of the target resolution and box-downsampled,
and an optional light bloom pass runs on each frame. For single
maximum-fidelity stills use blackhorizon.offline.render instead.

Run:
    python -m blackhorizon.offline.video --seconds 8 --output orbit.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy

from ..imaging import save_png
from ..realtime.settings import (
    BackgroundMode,
    QualityPreset,
    RenderSettings,
)
from .camera_path import CameraPath, orbit_path
from .post import add_bloom, encode_srgb


def downsample(image: numpy.ndarray, factor: int) -> numpy.ndarray:
    """Box-average an (h, w, 3) uint8 image by an integer factor."""
    if factor == 1:
        return image
    h, w = image.shape[0] // factor, image.shape[1] // factor
    view = image[: h * factor, : w * factor].astype(numpy.float32)
    return (
        view.reshape(h, factor, w, factor, 3)
        .mean(axis=(1, 3))
        .astype(numpy.uint8)
    )


def frame_bloom(image: numpy.ndarray, strength: float) -> numpy.ndarray:
    """Light bloom on a display-referred frame.

    The engine outputs tone-mapped LDR, so the frame is lifted to a
    pseudo-linear domain, bloomed, and re-encoded; an approximation
    that reads well in motion.
    """
    if strength <= 0.0:
        return image
    linear = (image.astype(numpy.float32) / 255.0) ** 2.2
    bloomed = add_bloom(
        linear, threshold=0.55, strength=strength, sigma=5.0
    )
    return encode_srgb(numpy.clip(bloomed, 0.0, 1.0))


def render_video(
    path: CameraPath,
    settings: RenderSettings,
    width: int,
    height: int,
    fps: int,
    output: str,
    supersample: int = 2,
    bloom_strength: float = 0.25,
    frames_dir: str | None = None,
    progress: bool = True,
) -> int:
    """Render a camera path to an H.264 video.

    Args:
        path: The camera path; its duration sets the video length.
        settings: Real-time render settings (spin, disk, quality).
        width: Output width in pixels.
        height: Output height in pixels.
        fps: Frames per second.
        output: Output .mp4 path.
        supersample: Integer render-scale multiple for antialiasing.
        bloom_strength: Per-frame bloom; 0 disables.
        frames_dir: Optional directory to also keep the PNG frames.
        progress: Print progress lines.

    Returns:
        The number of frames rendered.
    """
    import moderngl

    try:
        import imageio
    except ImportError as error:
        raise RuntimeError(
            "video encoding needs the 'video' extra: "
            'pip install -e ".[video]"'
        ) from error

    from ..realtime.engine import KerrRenderEngine

    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
    except Exception:
        ctx = moderngl.create_context(standalone=True)
    engine = KerrRenderEngine(ctx)
    keep_dir = pathlib.Path(frames_dir) if frames_dir else None
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)

    n_frames = max(2, int(round(path.duration * fps)))
    writer = imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8
    )
    start = time.perf_counter()
    try:
        for index in range(n_frames):
            t = index / fps
            pose = path.pose_at(t)
            settings.fov_degrees = pose.fov_degrees
            camera = path.camera_at(t)
            frame = engine.read_frame(
                settings,
                camera,
                width * supersample,
                height * supersample,
            )
            frame = downsample(frame, supersample)
            frame = frame_bloom(frame, bloom_strength)
            writer.append_data(frame)
            if keep_dir:
                save_png(frame, str(keep_dir / f"frame_{index:05d}.png"))
            if progress and (index % max(1, n_frames // 10) == 0):
                elapsed = time.perf_counter() - start
                print(
                    f"  frame {index + 1}/{n_frames} ({elapsed:.0f} s)"
                )
    finally:
        writer.close()
        engine.release()
        ctx.release()
    return n_frames


def encode_frames(pattern_dir: str, output: str, fps: int) -> int:
    """Encode an existing directory of PNG frames to an mp4.

    Frames are taken in sorted filename order, so zero-padded indices
    (like the TDE demo writes) encode correctly.
    """
    try:
        import imageio
    except ImportError as error:
        raise RuntimeError(
            "video encoding needs the 'video' extra: "
            'pip install -e ".[video]"'
        ) from error

    frames = sorted(pathlib.Path(pattern_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no PNG frames in {pattern_dir}")
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(imageio.v3.imread(frame))
    finally:
        writer.close()
    return len(frames)


def main() -> None:
    """Command line entry point for orbit videos and frame encoding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin", type=float, default=0.9)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=26.0)
    parser.add_argument("--inclination", type=float, default=82.0)
    parser.add_argument("--revolutions", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument(
        "--quality",
        choices=("low", "medium", "high", "ultra"),
        default="high",
    )
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--bloom-strength", type=float, default=0.25)
    parser.add_argument("--no-disk", action="store_true")
    parser.add_argument("--disk-outer", type=float, default=18.0)
    parser.add_argument("--disk-temperature", type=float, default=6500.0)
    parser.add_argument("--exposure", type=float, default=1.5)
    parser.add_argument("--frames-dir", type=str, default="")
    parser.add_argument(
        "--encode-frames",
        type=str,
        default="",
        help="skip rendering; encode PNG frames from this directory",
    )
    parser.add_argument("--output", type=str, default="blackhorizon.mp4")
    args = parser.parse_args()

    if args.encode_frames:
        count = encode_frames(args.encode_frames, args.output, args.fps)
        print(f"Encoded {count} frames to {args.output}")
        return

    settings = RenderSettings(
        spin=args.spin,
        fov_degrees=args.fov,
        resolution_scale=1.0,
        background=BackgroundMode.STARFIELD,
        disk_enabled=not args.no_disk,
        disk_outer_radius=args.disk_outer,
        disk_temperature=args.disk_temperature,
        exposure=args.exposure,
    ).apply_preset(QualityPreset(args.quality))
    settings.resolution_scale = 1.0

    path = orbit_path(
        args.seconds,
        args.distance,
        args.inclination,
        args.revolutions,
        args.fov,
    )
    start = time.perf_counter()
    count = render_video(
        path,
        settings,
        args.width,
        args.height,
        args.fps,
        args.output,
        supersample=args.supersample,
        bloom_strength=args.bloom_strength,
        frames_dir=args.frames_dir or None,
    )
    print(
        f"Rendered {count} frames at {args.width}x{args.height} "
        f"in {time.perf_counter() - start:.0f} s, wrote {args.output}"
    )


if __name__ == "__main__":
    main()
