"""
Global 4D phase-field collective dynamics simulation.

Key modeling choices (per request):
1) Use a *global* phase-field coupling via the order parameter:
     O = <exp(i*phi)>  (spatial average)
2) Implement "momentum transfer" consistent with:
     p = ħ * k
   by applying an external kick in Fourier space proportional to |k|.

This is a lightweight, self-contained research scaffold intended for
experimenting with qualitative dynamics and topology diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _wrap_to_2pi(phi: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 2π)."""
    twopi = 2.0 * np.pi
    return np.mod(phi, twopi)


def _order_parameter(phi: np.ndarray) -> complex:
    """Global Kuramoto-like order parameter."""
    return np.mean(np.exp(1j * phi))


@dataclass(frozen=True)
class PhaseSimConfig:
    # Lattice size per dimension (4D hypercubic grid)
    L: int = 8

    # Time integration
    dt: float = 0.05
    steps: int = 800
    save_every: int = 5

    # Model terms
    coupling_global: float = 1.0  # alignment strength via order parameter
    laplacian_strength: float = 0.25  # κ * ∇^2 φ term (4D Laplacian)
    potential_strength: float = 0.35  # V(φ) = A*(1-cos(φ))
    potential_phase: float = 0.0  # shift in cosine potential

    # Damping/noise
    damping: float = 0.12  # -η p
    noise_strength: float = 0.0  # additive noise on momentum equation

    # Effective constants for the "p = ħ k" kick
    hbar: float = 1.0
    drive_amplitude: float = 0.25  # A0(t) amplitude
    drive_omega: float = 1.2  # angular frequency
    drive_phase: float = 0.0  # phase offset
    drive_sigma: float = 0.9  # momentum transfer bandwidth in k-space
    drive_k_mag: Optional[float] = None  # if None, inferred from dominant k magnitude

    # Randomness
    seed: int = 0

    # Numerical stability
    enforce_wrap_phi: bool = True


@dataclass
class PhaseSimState:
    phi: np.ndarray  # real phase field, shape (L,L,L,L)
    p: np.ndarray  # real momentum field (conjugate to phi), same shape


def _make_k_grids(L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return physical wavevectors kx,ky,kz,kw on the FFT grid.

    We use FFT frequency bins and multiply by 2π so that:
      exp(i k x) is consistent with angular phase conventions.
    """
    k1d = 2.0 * np.pi * np.fft.fftfreq(L, d=1.0)
    kx, ky, kz, kw = np.meshgrid(k1d, k1d, k1d, k1d, indexing="ij")
    return kx, ky, kz, kw


def _laplacian_4d(phi: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Compute 4D Laplacian using FFT: ∇^2 φ."""
    phi_hat = np.fft.fftn(phi)
    lap_hat = -(k2) * phi_hat
    lap = np.fft.ifftn(lap_hat)
    return lap.real


def _kick_force_p_dot(
    t: float,
    config: PhaseSimConfig,
    k_mag: np.ndarray,
    k2: np.ndarray,
) -> np.ndarray:
    """
    Momentum transfer kick: dp/dt contribution in real space.

    We apply a Fourier-space forcing proportional to |k| (so the kick respects
    p = ħ k at the mode level) with a Gaussian bandwidth around drive_k_mag.
    """
    A_t = config.drive_amplitude * np.sin(config.drive_omega * t + config.drive_phase)
    if config.drive_k_mag is None:
        # Default: kick near the largest nonzero |k| present on the discrete grid.
        # This provides nontrivial dynamics even without explicit tuning.
        drive_center = float(np.max(k_mag))
    else:
        drive_center = float(config.drive_k_mag)

    # Gaussian selection in |k|
    kernel = np.exp(-0.5 * ((k_mag - drive_center) / max(config.drive_sigma, 1e-12)) ** 2)

    # Force in Fourier space (real kernel -> real force after ifft)
    kick_hat = (config.hbar * k_mag) * kernel * A_t
    kick = np.fft.ifftn(kick_hat)
    # Numerical noise may introduce small imaginary parts; keep real part.
    return kick.real


def initialize_state(config: PhaseSimConfig) -> PhaseSimState:
    rng = np.random.default_rng(config.seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(config.L, config.L, config.L, config.L)).astype(np.float64)
    p = np.zeros_like(phi)
    return PhaseSimState(phi=phi, p=p)


def evolve(
    state: PhaseSimState,
    config: PhaseSimConfig,
    *,
    record_phi: bool = False,
) -> Dict[str, Any]:
    """
    Run time integration and return history dict.

    Important diagnostics returned:
      - order_parameter: complex O(t)
      - order_magnitude: |O(t)|
      - phi_snapshots (optional)
    """
    L = config.L
    kx, ky, kz, kw = _make_k_grids(L)
    k2 = kx * kx + ky * ky + kz * kz + kw * kw
    k_mag = np.sqrt(k2)

    rng = np.random.default_rng(config.seed + 12345)

    phi = state.phi
    p = state.p

    times: List[float] = []
    order_complex: List[complex] = []
    order_magnitude: List[float] = []
    phi_snaps: List[np.ndarray] = []
    step_indices: List[int] = []

    def compute_force(phi_field: np.ndarray) -> np.ndarray:
        # Local potential term: V = A*(1-cos(phi - phi0))  => -dV/dphi = -A*sin(phi - phi0)
        local_force = -config.potential_strength * np.sin(phi_field - config.potential_phase)

        # Global coupling via order parameter O = <exp(i phi)>
        O = _order_parameter(phi_field)
        global_force = config.coupling_global * np.imag(O * np.exp(-1j * phi_field))

        # 4D Laplacian term
        lap_force = config.laplacian_strength * _laplacian_4d(phi_field, k2)

        return local_force + global_force + lap_force

    for n in range(config.steps):
        t = n * config.dt

        # First compute force at current time
        F = compute_force(phi)
        kick = _kick_force_p_dot(t, config=config, k_mag=k_mag, k2=k2)

        # Momentum half-step (velocity-Verlet / leapfrog style)
        p_half = p + 0.5 * config.dt * (F + kick - config.damping * p)

        # Phase full-step
        phi_new = phi + config.dt * p_half
        if config.enforce_wrap_phi:
            phi_new = _wrap_to_2pi(phi_new)

        # Next force at t + dt
        F2 = compute_force(phi_new)
        kick2 = _kick_force_p_dot(t + config.dt, config=config, k_mag=k_mag, k2=k2)
        if config.noise_strength > 0.0:
            # Noise on momentum equation: dp/dt includes noise_strength * ξ(t)
            eta = rng.normal(0.0, 1.0, size=phi.shape).astype(np.float64)
        else:
            eta = 0.0

        p_new = p_half + 0.5 * config.dt * (F2 + kick2 - config.damping * p_half) + (config.noise_strength * np.sqrt(config.dt) * eta)

        # Update state
        phi = phi_new
        p = p_new

        if (n % config.save_every) == 0 or n == config.steps - 1:
            O = _order_parameter(phi)
            times.append(t)
            order_complex.append(O)
            order_magnitude.append(float(np.abs(O)))
            step_indices.append(n)
            if record_phi:
                phi_snaps.append(phi.copy())

    return {
        "times": np.array(times, dtype=np.float64),
        "order_parameter": np.array(order_complex, dtype=np.complex128),
        "order_magnitude": np.array(order_magnitude, dtype=np.float64),
        "step_indices": np.array(step_indices, dtype=np.int64),
        "final_state": PhaseSimState(phi=phi, p=p),
        "phi_snapshots": phi_snaps if record_phi else None,
    }


def run_collective_dynamics(
    config: PhaseSimConfig,
    *,
    initial_state: Optional[PhaseSimState] = None,
    record_phi: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
      - initialize state (if needed)
      - evolve
    """
    if initial_state is None:
        initial_state = initialize_state(config)
    return evolve(initial_state, config, record_phi=record_phi)

