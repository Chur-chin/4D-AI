"""
Graphene-hBN material calibration utilities.

This module provides a *numerical scaffold* that maps commonly referenced
materials parameters (twist angle, dielectric screening, etc.) into the
dimensionless parameters used by `phase_collective_sim.py`.

It is intentionally lightweight: the goal is to keep the code runnable and
experiment-friendly for qualitative 4D phase-field dynamics (global coupling
and p = ħ k momentum-transfer kicks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class GrapheneHBNInputs:
    twist_angle_deg: float = 1.5
    # Dielectric environment (relative permittivity)
    eps_hbn: float = 4.5
    eps_env: float = 2.5  # encapsulating environment (e.g. vacuum/encapsulation)

    # Temperature affects effective damping/noise scale in this scaffold
    temperature_K: float = 4.0

    # Optional tuning: overall scale to adapt to your preferred unit system
    scale: float = 1.0

    # Lattice constant for moiré length estimate (nm), typical graphene value
    graphene_a0_nm: float = 0.246


def moire_length_nm(twist_angle_deg: float, a0_nm: float = 0.246) -> float:
    """
    Approximate moiré length for graphene-hBN:
      Lm ≈ a0 / (2 sin(θ/2))
    """
    theta = np.deg2rad(float(twist_angle_deg))
    if np.isclose(theta, 0.0):
        return float("inf")
    return float(a0_nm / (2.0 * np.sin(theta / 2.0)))


def effective_dielectric(eps_hbn: float, eps_env: float) -> float:
    """Simple average-screening model used for dimensionless parameter scaling."""
    return float(0.5 * (eps_hbn + eps_env))


def calibrate_graphene_hbn(inputs: GrapheneHBNInputs) -> Dict[str, float]:
    """
    Map material inputs -> simulation parameters.

    Returned keys are meant to be used as overrides for PhaseSimConfig.
    """
    Lm = moire_length_nm(inputs.twist_angle_deg, inputs.graphene_a0_nm)
    eps_eff = effective_dielectric(inputs.eps_hbn, inputs.eps_env)

    # Convert moiré length into a dimensionless "stiffness" scale.
    # Smaller moiré periods -> stronger gradients/coupling in this scaffold.
    if not np.isfinite(Lm) or Lm <= 0:
        stiffness = 1.0
    else:
        stiffness = 1.0 / Lm

    # Dimensionless screening: larger epsilon => weaker effective coupling.
    screening = 1.0 / max(eps_eff, 1e-6)

    # Temperature -> damping increase (very rough scaffold).
    temp_factor = 1.0 + 0.02 * max(inputs.temperature_K, 0.0)

    scale = float(inputs.scale)

    # Base values are tuned to produce stable, nontrivial dynamics for default grid sizes.
    coupling_global = scale * (0.9 * screening * stiffness) / (stiffness + 0.5)
    laplacian_strength = scale * (0.6 * stiffness * stiffness) * 0.25
    potential_strength = scale * (0.7 * screening) * 0.25

    damping = scale * (0.08 + 0.06 * np.tanh(temp_factor - 1.0)) + 0.02

    # Drive bandwidth: roughly proportional to inverse moiré length
    drive_sigma = float(np.clip(0.5 + 1.2 * stiffness, 0.25, 2.5))

    return {
        "coupling_global": float(coupling_global),
        "laplacian_strength": float(laplacian_strength),
        "potential_strength": float(potential_strength),
        "damping": float(damping),
        "drive_sigma": float(drive_sigma),
        # Keep ħ = 1 by default; users can rescale separately.
        "hbar": 1.0,
    }


def apply_calibration_to_config(
    base_config: "Optional[object]",
    calib: Dict[str, float],
    *,
    require_fields: bool = True,
) -> Dict[str, float]:
    """
    Convert a calibration dict into config overrides.

    We return overrides rather than mutating the config to keep this module
    decoupled from `phase_collective_sim.py` implementation details.
    """
    overrides: Dict[str, float] = {}
    for k, v in calib.items():
        if require_fields and base_config is not None:
            if not hasattr(base_config, k):
                continue
        overrides[k] = float(v)
    return overrides

