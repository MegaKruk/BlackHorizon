"""Blackbody color synthesis.

Maps temperatures to sRGB colors by integrating the Planck spectrum
against the CIE 1931 color matching functions (multi-Gaussian fits of
Wyman, Sloan and Shirley 2013), converting XYZ to linear sRGB, and
normalizing each color so its largest channel is one. Brightness is
deliberately excluded: it scales as T^4 by Stefan-Boltzmann and is
applied separately by the renderer, keeping the lookup purely chromatic.
"""

from __future__ import annotations

import numpy

# Planck constants in SI units.
_H = 6.62607015e-34
_C = 2.99792458e8
_K_B = 1.380649e-23

_XYZ_TO_SRGB = numpy.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)

TEMPERATURE_MIN = 500.0
TEMPERATURE_MAX = 40000.0


def _piecewise_gaussian(x, mean, sigma_low, sigma_high):
    sigma = numpy.where(x < mean, sigma_low, sigma_high)
    return numpy.exp(-0.5 * ((x - mean) / sigma) ** 2)


def color_matching_functions(wavelengths_nm):
    """CIE 1931 xbar, ybar, zbar (Wyman et al. 2013 Gaussian fits)."""
    w = wavelengths_nm
    xbar = (
        1.056 * _piecewise_gaussian(w, 599.8, 37.9, 31.0)
        + 0.362 * _piecewise_gaussian(w, 442.0, 16.0, 26.7)
        - 0.065 * _piecewise_gaussian(w, 501.1, 20.4, 26.2)
    )
    ybar = 0.821 * _piecewise_gaussian(
        w, 568.8, 46.9, 40.5
    ) + 0.286 * _piecewise_gaussian(w, 530.9, 16.3, 31.1)
    zbar = 1.217 * _piecewise_gaussian(
        w, 437.0, 11.8, 36.0
    ) + 0.681 * _piecewise_gaussian(w, 459.0, 26.0, 13.8)
    return xbar, ybar, zbar


def planck_spectral_radiance(wavelengths_nm, temperature):
    """Planck spectral radiance per wavelength (arbitrary common scale)."""
    lam = wavelengths_nm * 1e-9
    exponent = _H * _C / (lam * _K_B * temperature)
    exponent = numpy.minimum(exponent, 700.0)
    return 1.0 / (lam**5 * numpy.expm1(exponent))


def blackbody_rgb(temperature: float) -> numpy.ndarray:
    """Normalized linear sRGB color of a blackbody at a temperature.

    Args:
        temperature: Temperature in Kelvin; clipped to the supported
            range [500, 40000].

    Returns:
        Linear sRGB triple with the largest channel equal to one.
    """
    t = float(numpy.clip(temperature, TEMPERATURE_MIN, TEMPERATURE_MAX))
    wavelengths = numpy.linspace(380.0, 780.0, 200)
    spectrum = planck_spectral_radiance(wavelengths, t)
    xbar, ybar, zbar = color_matching_functions(wavelengths)
    xyz = numpy.array(
        [
            numpy.trapezoid(spectrum * xbar, wavelengths),
            numpy.trapezoid(spectrum * ybar, wavelengths),
            numpy.trapezoid(spectrum * zbar, wavelengths),
        ]
    )
    rgb = _XYZ_TO_SRGB @ xyz
    rgb = numpy.maximum(rgb, 0.0)
    peak = rgb.max()
    if peak <= 0.0:
        return numpy.array([1.0, 1.0, 1.0])
    return rgb / peak


def blackbody_lut(size: int = 256) -> tuple[numpy.ndarray, float, float]:
    """Chromaticity lookup table over log-spaced temperatures.

    Args:
        size: Number of table entries.

    Returns:
        Tuple (table, log_t_min, log_t_max) where table has shape
        (size, 3) in linear sRGB and the log bounds define the sampling
        coordinate u = (log(T) - log_t_min) / (log_t_max - log_t_min).
    """
    log_min = float(numpy.log(TEMPERATURE_MIN))
    log_max = float(numpy.log(TEMPERATURE_MAX))
    temperatures = numpy.exp(numpy.linspace(log_min, log_max, size))
    table = numpy.stack([blackbody_rgb(t) for t in temperatures])
    return table.astype(numpy.float32), log_min, log_max
