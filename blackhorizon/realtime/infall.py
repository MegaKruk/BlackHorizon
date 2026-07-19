"""Doomed-observer camera: a proper-time geodesic with thrust.

Once the camera crosses the event horizon the app hands it to this
state machine. The camera worldline is a timelike geodesic integrated
in proper time; the user can still look around freely and fire
thrusters, but inside the horizon the radial coordinate is timelike,
so no thrust can increase r; burns only reshape the plunge and, away
from the maximal-lifetime trajectory, shorten it (Lewis and Kwan 2007,
arXiv:0705.1029). The countdown reports the proper time remaining to
the terminal radius along the current worldline if the engines stay
cold, recomputed by lookahead integration after every burn.
"""

from __future__ import annotations

import math

import numpy

from ..frames import build_tetrad, lower_index, rain_four_velocity, raise_index
from ..geodesics import build_state, geodesic_rhs
from ..integrators import rk4_step
from ..kerr import KerrSpacetime


class InfallState:
    """Timelike camera worldline inside the horizon.

    Attributes:
        spacetime: The Kerr spacetime being fallen through.
        state: Geodesic state of shape (1, 8) in proper-time gauge.
        stop_radius: Radius where the worldline terminates.
        remaining_tau: Cached coasting proper time to the stop radius.
        elapsed_tau: Proper time since horizon crossing.
    """

    def __init__(
        self,
        spacetime: KerrSpacetime,
        state: numpy.ndarray,
        stop_radius: float,
        journey: str = "realistic",
    ) -> None:
        self.spacetime = spacetime
        self.base_stop_radius = float(stop_radius)
        self.journey = journey
        self.stop_radius = self._terminal_radius(journey)
        self.state = state
        self.elapsed_tau = 0.0
        self._forced_termination = False
        self.remaining_tau = self.lookahead_remaining_tau()

    def _terminal_radius(self, journey: str) -> float:
        """Terminal surface for the chosen journey mode.

        Realistic: the Cauchy horizon for spinning holes (the blue
        sheet ends the infall there). Idealized: the requested stop
        radius, continuing into the inner region of exact vacuum Kerr.
        """
        inner = float(
            getattr(self.spacetime, "inner_horizon_radius", 0.0)
        )
        if journey == "idealized":
            return self.base_stop_radius
        return max(self.base_stop_radius, inner * 1.02)

    def set_journey(self, journey: str) -> None:
        """Switch journey mode mid-flight; recomputes the countdown."""
        if journey == self.journey:
            return
        self.journey = journey
        self.stop_radius = self._terminal_radius(journey)
        if not self.terminated():
            self.remaining_tau = self.lookahead_remaining_tau()

    @classmethod
    def from_crossing(
        cls,
        spacetime: KerrSpacetime,
        position: numpy.ndarray,
        stop_radius: float,
        journey: str = "realistic",
    ) -> "InfallState":
        """Start an infall at the given position on the rain worldline.

        The rain (Doran) 4-velocity is the physical free-fall-from-rest
        state and is regular at the crossing; the user's subsequent
        burns take the worldline anywhere the physics allows.
        """
        pos = numpy.asarray(position, dtype=float)[None, :]
        velocity = rain_four_velocity(spacetime, pos)
        momentum = lower_index(spacetime, pos, velocity)
        return cls(
            spacetime, build_state(pos, momentum), stop_radius, journey
        )

    @property
    def position(self) -> numpy.ndarray:
        """Current spatial position, shape (3,)."""
        return self.state[0, 1:4].copy()

    @property
    def radius(self) -> float:
        """Current Kerr-Schild radius."""
        return float(
            self.spacetime.kerr_schild_radius(
                self.state[:, 1], self.state[:, 2], self.state[:, 3]
            )[0]
        )

    @property
    def four_velocity(self) -> numpy.ndarray:
        """Contravariant camera 4-velocity, shape (4,)."""
        return raise_index(
            self.spacetime, self.state[:, 1:4], self.state[:, 4:8]
        )[0]

    def terminated(self) -> bool:
        """Whether the worldline has reached the terminal radius."""
        return self._forced_termination or self.radius <= self.stop_radius

    @property
    def termination_reason(self) -> str:
        """Why the worldline ended: surface, chart, or none.

        "surface": reached the terminal radius (the singularity stop,
        or the Cauchy horizon in realistic mode). "chart": the state
        left the well-conditioned region of the single stationary
        Kerr-Schild chart, which in idealized mode happens at the ring
        plane (the gateway to negative r) or when an outgoing branch
        asymptotes the Cauchy horizon trying to exit into the next
        universe of the maximal extension.
        """
        if not self.terminated():
            return "none"
        if self.radius <= self.stop_radius * 1.5:
            return "surface"
        return "chart"

    def _renormalize(self) -> bool:
        """Restore the timelike mass shell against integration drift.

        The Hamiltonian is exactly -1/2 on the worldline; stiff fields
        near the terminal radius can drift it. Rescaling the covariant
        momentum by sqrt(1 / (-2H)) restores the shell exactly while
        preserving the direction. Returns False when the state is
        beyond repair (H >= 0 or non-finite), which forces termination.
        """
        from ..geodesics import hamiltonian

        value = float(hamiltonian(self.spacetime, self.state)[0])
        if not numpy.isfinite(value) or value >= -1e-6:
            return False
        if abs(value + 0.5) > 1e-9:
            self.state[:, 4:8] *= math.sqrt(0.5 / (-value))
        return True

    def _substep(self, radius: float, budget: float) -> float:
        """Radius-adaptive proper-time substep.

        Near the center the plunge speed grows as sqrt(2M/r), so a
        step proportional to r^(3/2) keeps the radial advance per step
        a fixed fraction of r; capped for the smooth outer region.
        """
        return float(
            min(budget, 0.004, max(0.04 * radius**1.5, 1e-6))
        )

    def advance(self, dtau: float) -> None:
        """Advance the worldline by proper time dtau."""
        if dtau <= 0.0 or self.terminated():
            return

        def rhs(batch):
            return geodesic_rhs(self.spacetime, batch)

        remaining = dtau
        while remaining > 0.0 and not self.terminated():
            previous = self.state.copy()
            h_val = self._substep(self.radius, remaining)
            self.state = rk4_step(
                rhs, self.state, numpy.array([h_val])
            )
            self.elapsed_tau += h_val
            remaining -= h_val
            if not numpy.isfinite(self.state).all() or not (
                self._renormalize()
            ):
                # Safety net: keep the last well-conditioned state so
                # the camera and tetrad stay valid, and end the
                # worldline there.
                self.state = previous
                self._forced_termination = True
                break
        self.remaining_tau = max(0.0, self.remaining_tau - dtau)

    def thrust(
        self,
        local_direction: numpy.ndarray,
        rapidity: float,
        forward: numpy.ndarray,
        up: numpy.ndarray,
        recompute_lookahead: bool = True,
    ) -> None:
        """Fire thrusters: an exact boost in the camera's local frame.

        The new 4-velocity is cosh(a) e0 + sinh(a) d with d the unit
        combination of the spatial tetrad legs along local_direction,
        preserving the timelike normalization exactly. Inside the
        horizon this can never increase r; it only reshapes the plunge.

        Args:
            local_direction: Burn direction in the local frame
                (components along forward, right, up), shape (3,).
            rapidity: Boost rapidity of this burn.
            forward: Current view direction seed for the tetrad.
            up: Current up seed for the tetrad.
        """
        norm = float(numpy.linalg.norm(local_direction))
        if norm < 1e-12 or rapidity == 0.0 or self.terminated():
            return
        direction = numpy.asarray(local_direction, dtype=float) / norm
        position = self.position
        tetrad = build_tetrad(
            self.spacetime, position, self.four_velocity, forward, up
        )
        spatial = (
            direction[0] * tetrad[1]
            + direction[1] * tetrad[2]
            + direction[2] * tetrad[3]
        )
        boosted = (
            math.cosh(rapidity) * tetrad[0]
            + math.sinh(rapidity) * spatial
        )
        momentum = lower_index(
            self.spacetime, position[None, :], boosted[None, :]
        )
        self.state[:, 4:8] = momentum
        if recompute_lookahead:
            self.remaining_tau = self.lookahead_remaining_tau()

    def lookahead_remaining_tau(
        self, max_tau: float = 60.0, step: float = 0.001
    ) -> float:
        """Coasting proper time from here to the stop radius.

        Integrates a copy of the current state with cold engines; the
        physics guarantees the plunge ends, so the budget is a safety
        net only.
        """
        probe = self.state.copy()

        def rhs(batch):
            return geodesic_rhs(self.spacetime, batch)

        tau = 0.0
        while tau < max_tau:
            radius = float(
                self.spacetime.kerr_schild_radius(
                    probe[:, 1], probe[:, 2], probe[:, 3]
                )[0]
            )
            if radius <= self.stop_radius or not numpy.isfinite(
                probe
            ).all():
                return tau
            h_val = self._substep(radius, step * 4.0)
            probe = rk4_step(rhs, probe, numpy.array([h_val]))
            tau += h_val
        return max_tau
