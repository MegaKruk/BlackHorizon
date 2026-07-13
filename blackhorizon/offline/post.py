"""HDR post-processing for the offline renderer.

Operates on linear-light HDR images (float arrays, values in [0, inf))
and produces display-ready 8-bit sRGB. The pipeline is bloom (a bright
pass blurred with a separable Gaussian and added back, the glow that
makes hot disk regions bleed like camera optics), the ACES filmic tone
curve (Narkowicz 2015 fit), and sRGB gamma encoding. Pure NumPy.
"""

from __future__ import annotations

import numpy


def gaussian_kernel(sigma: float) -> numpy.ndarray:
    """Normalized 1D Gaussian kernel truncated at three sigma."""
    radius = max(1, int(numpy.ceil(3.0 * sigma)))
    x = numpy.arange(-radius, radius + 1, dtype=float)
    kernel = numpy.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(image: numpy.ndarray, sigma: float) -> numpy.ndarray:
    """Separable Gaussian blur of an (h, w, 3) image with edge padding."""
    kernel = gaussian_kernel(sigma)
    radius = kernel.size // 2
    padded = numpy.pad(image, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    blurred = numpy.zeros_like(image)
    for offset, weight in enumerate(kernel):
        blurred += weight * padded[offset : offset + image.shape[0]]
    padded = numpy.pad(
        blurred, ((0, 0), (radius, radius), (0, 0)), mode="edge"
    )
    result = numpy.zeros_like(image)
    for offset, weight in enumerate(kernel):
        result += weight * padded[:, offset : offset + image.shape[1]]
    return result


def add_bloom(
    hdr: numpy.ndarray,
    threshold: float = 1.0,
    strength: float = 0.35,
    sigma: float = 6.0,
) -> numpy.ndarray:
    """Add a blurred bright pass back onto the image.

    Args:
        hdr: Linear HDR image, shape (h, w, 3).
        threshold: Luminance above which pixels contribute to the glow.
        strength: Multiplier on the blurred bright pass.
        sigma: Gaussian blur width in pixels.

    Returns:
        HDR image with bloom, same shape.
    """
    luminance = (
        0.2126 * hdr[:, :, 0] + 0.7152 * hdr[:, :, 1] + 0.0722 * hdr[:, :, 2]
    )
    excess = numpy.maximum(luminance - threshold, 0.0)
    scale = numpy.where(luminance > 1e-9, excess / (luminance + 1e-9), 0.0)
    bright = hdr * scale[:, :, None]
    return hdr + strength * gaussian_blur(bright, sigma)


def aces_tonemap(hdr: numpy.ndarray) -> numpy.ndarray:
    """ACES filmic tone curve (Narkowicz 2015 fit), output in [0, 1]."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e)
    return numpy.clip(mapped, 0.0, 1.0)


def encode_srgb(linear: numpy.ndarray) -> numpy.ndarray:
    """Linear [0, 1] to 8-bit sRGB with the standard piecewise gamma."""
    linear = numpy.clip(linear, 0.0, 1.0)
    encoded = numpy.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * numpy.power(linear, 1.0 / 2.4) - 0.055,
    )
    return (encoded * 255.0 + 0.5).astype(numpy.uint8)


def develop(
    hdr: numpy.ndarray,
    exposure: float = 1.0,
    bloom_threshold: float = 1.0,
    bloom_strength: float = 0.35,
    bloom_sigma: float = 6.0,
) -> numpy.ndarray:
    """Full development: exposure, bloom, ACES, sRGB encode.

    Args:
        hdr: Linear HDR image, shape (h, w, 3).
        exposure: Linear multiplier applied before everything else.
        bloom_threshold: See add_bloom; set strength to 0 to disable.
        bloom_strength: See add_bloom.
        bloom_sigma: See add_bloom.

    Returns:
        Display-ready uint8 image, shape (h, w, 3).
    """
    image = hdr * exposure
    if bloom_strength > 0.0:
        image = add_bloom(image, bloom_threshold, bloom_strength, bloom_sigma)
    return encode_srgb(aces_tonemap(image))
