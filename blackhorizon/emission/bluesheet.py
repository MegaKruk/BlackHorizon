"""Blue-sheet physics at the inner (Cauchy) horizon.

A real black hole is illuminated forever: starlight, the cosmic
microwave background, and its own accretion disk keep entering at all
advanced times v. Light that entered at advanced time v reaches an
observer approaching the Cauchy horizon amplified by exp(kappa_minus
v), where kappa_minus is the inner-horizon surface gravity; the pileup
of this radiation is the blue sheet, whose backreaction drives mass
inflation (Poisson and Israel, Phys. Rev. D 41, 1796 (1990); Ori,
Phys. Rev. Lett. 67, 789 (1991); Hamilton and Avelino,
arXiv:0811.1926). Dafermos and Luk (arXiv:1710.01722) prove the
metric nevertheless extends continuously to the Cauchy horizon, so
the exact-Kerr geometry, aberration, and covariant per-ray shift
factors used everywhere else in this simulator remain valid along the
whole approach; the blue sheet enters purely as a multiplicative
amplification of received radiation.

The closed-form law used here: along an infalling worldline the
advanced time to the sheet obeys v = -(1/kappa_minus) ln x + const
with proximity x = (r - r_minus)/(r_plus - r_minus), the standard
near-horizon relation. For steady external illumination the received
amplification of every external ray is therefore

    B(r) = x_match / x        for x < x_match, else 1,

continuous at the matching proximity and diverging as 1/x at the
sheet; received intensity scales as B^4 (Liouville invariance of
I_nu / nu^3 integrated over frequency). The angular structure of the
flare is not modeled here: it comes for free from the exact geodesic
sky mapping that B multiplies. In idealized journey mode (eternal
vacuum Kerr, no infalling radiation) the amplification is identically
one, which keeps that mode honest.

All routines are array-module generic and safe at zero spin, where no
inner horizon exists and every amplification is one.
"""

from __future__ import annotations

from ..backend import Array, xp_of
from ..kerr import KerrSpacetime

DEFAULT_MATCHING_PROXIMITY = 0.5
DEFAULT_AMPLIFICATION_CAP = 60.0


def inner_surface_gravity(spacetime: KerrSpacetime) -> float:
    """Surface gravity kappa_minus of the inner horizon.

    kappa_minus = (r_plus - r_minus) / (2 (r_minus^2 + a^2)), the
    e-folding rate of the blue-sheet amplification in advanced time.
    Returns zero for a Schwarzschild hole, which has no inner horizon.
    """
    r_plus = float(spacetime.outer_horizon_radius)
    r_minus = float(spacetime.inner_horizon_radius)
    if r_minus <= 0.0:
        return 0.0
    spin = float(spacetime.spin) * float(spacetime.mass)
    return (r_plus - r_minus) / (2.0 * (r_minus**2 + spin**2))


def proximity(spacetime: KerrSpacetime, radius: Array) -> Array:
    """Normalized distance x to the inner horizon, clamped to (0, 1].

    x = (r - r_minus) / (r_plus - r_minus); the floor keeps the
    amplification finite at the terminal surface itself.
    """
    xp = xp_of(radius)
    r_plus = float(spacetime.outer_horizon_radius)
    r_minus = float(spacetime.inner_horizon_radius)
    if r_minus <= 0.0:
        return xp.ones_like(radius)
    span = max(r_plus - r_minus, 1e-12)
    return xp.clip((radius - r_minus) / span, 1e-6, 1.0)


def blueshift_amplification(
    spacetime: KerrSpacetime,
    radius: Array,
    matching_proximity: float = DEFAULT_MATCHING_PROXIMITY,
    cap: float = DEFAULT_AMPLIFICATION_CAP,
) -> Array:
    """Blue-sheet amplification B(r) of received external radiation.

    One away from the inner horizon, x_match / x inside the matching
    proximity, capped for numerical sanity (the cap stands in for the
    finite duration of any real approach; intensity at the cap is
    already cap^4 times ambient). Identically one at zero spin.

    Args:
        spacetime: The Kerr spacetime.
        radius: Kerr-Schild radii, any shape.
        matching_proximity: Proximity below which amplification grows.
        cap: Maximum amplification.

    Returns:
        Amplification factors, same shape as radius.
    """
    xp = xp_of(radius)
    if float(spacetime.inner_horizon_radius) <= 0.0:
        return xp.ones_like(radius)
    x = proximity(spacetime, radius)
    return xp.clip(matching_proximity / x, 1.0, cap)


def sheet_radiance(
    amplification: Array, base_intensity: float = 0.02
) -> Array:
    """HDR radiance of the blue sheet itself for terminated rays.

    Rays that end on the inner-horizon terminal surface are looking
    into the infalling radiation pileup; their radiance follows the
    same B^4 law from a faint ambient base, so the wall fades in
    smoothly and saturates the tone mapper only on close approach.
    """
    return base_intensity * (amplification - 1.0) ** 4


SHEET_COLOR = (0.62, 0.78, 1.0)

DISPLAY_CAP = 8.0
WHITEOUT_START = 8.0
WHITEOUT_END = 30.0


def display_amplification(amplification: Array) -> Array:
    """Photographically adapted amplification for rendering.

    The physical amplification diverges at the sheet; a display,
    like any eye or camera, adapts exposure. Capping the intensity
    gain at DISPLAY_CAP (a 4096-fold brightening) preserves per-ray
    structure through the approach while the HUD reports the true
    value; the terminal saturation is handed to the whiteout ramp.
    """
    xp = xp_of(amplification)
    return xp.clip(amplification, 1.0, DISPLAY_CAP)


def whiteout_fraction(amplification: Array) -> Array:
    """Fraction of the view overwhelmed by the radiation bath.

    Smoothly ramps from zero at WHITEOUT_START to near one at
    WHITEOUT_END: past the adapted display range, the blue sheet
    fills the sky from every direction and the frame washes toward
    white, the visually honest terminal state of the realistic
    journey.
    """
    xp = xp_of(amplification)
    t = xp.clip(
        (amplification - WHITEOUT_START)
        / (WHITEOUT_END - WHITEOUT_START),
        0.0,
        1.0,
    )
    return 0.92 * t * t * (3.0 - 2.0 * t)
