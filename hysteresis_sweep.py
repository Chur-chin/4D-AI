"""
Hysteresis sweep for the global 4D phase-field dynamics.

The hysteresis protocol:
1) Sweep drive amplitude A0 upward, carrying the final simulation state
   forward to the next A0.
2) Sweep A0 downward, again carrying states.

The returned hysteresis curve is based on the order parameter magnitude:
  |O(t)| = |<exp(i*phi)>|
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phase_collective_sim import PhaseSimConfig, PhaseSimState, run_collective_dynamics


@dataclass(frozen=True)
class HysteresisSweepConfig:
    A_values: Sequence[float]
    steps_per_point: int = 250
    save_every: int = 10
    record_phi: bool = False


def _run_point(
    base_config: PhaseSimConfig,
    A0: float,
    *,
    steps_per_point: int,
    save_every: int,
    initial_state: PhaseSimState,
    record_phi: bool,
) -> Tuple[Dict[str, Any], PhaseSimState]:
    cfg = replace(
        base_config,
        drive_amplitude=float(A0),
        steps=int(steps_per_point),
        save_every=int(max(1, save_every)),
    )
    result = run_collective_dynamics(cfg, initial_state=initial_state, record_phi=record_phi)
    final_state = result["final_state"]
    return result, final_state


def hysteresis_sweep(
    base_config: PhaseSimConfig,
    sweep_cfg: HysteresisSweepConfig,
    *,
    seed_initial_state: int = 0,
) -> Dict[str, Any]:
    """
    Return forward and backward hysteresis curves.

    Output fields:
      - forward: {"A0": [...], "order_end": [...], "order_max": [...]}
      - backward: same structure
    """
    A_values = list(map(float, sweep_cfg.A_values))
    if len(A_values) < 2:
        raise ValueError("A_values must have at least 2 elements")

    # Initial state for the upward sweep
    cfg0 = replace(base_config, seed=int(seed_initial_state))
    initial_state = PhaseSimState(
        phi=np.random.default_rng(cfg0.seed).uniform(0.0, 2.0 * np.pi, size=(cfg0.L, cfg0.L, cfg0.L, cfg0.L)).astype(np.float64),
        p=np.zeros((cfg0.L, cfg0.L, cfg0.L, cfg0.L), dtype=np.float64),
    )

    # Upward sweep
    forward_A: List[float] = []
    forward_order_end: List[float] = []
    forward_order_max: List[float] = []

    state = initial_state
    for A0 in A_values:
        result, state = _run_point(
            cfg0,
            A0,
            steps_per_point=sweep_cfg.steps_per_point,
            save_every=sweep_cfg.save_every,
            initial_state=state,
            record_phi=sweep_cfg.record_phi,
        )
        forward_A.append(A0)
        forward_order_end.append(float(result["order_magnitude"][-1]))
        forward_order_max.append(float(np.max(result["order_magnitude"])))

    # Downward sweep (start from the upward final state)
    backward_A: List[float] = []
    backward_order_end: List[float] = []
    backward_order_max: List[float] = []

    for A0 in reversed(A_values):
        result, state = _run_point(
            cfg0,
            A0,
            steps_per_point=sweep_cfg.steps_per_point,
            save_every=sweep_cfg.save_every,
            initial_state=state,
            record_phi=sweep_cfg.record_phi,
        )
        backward_A.append(A0)
        backward_order_end.append(float(result["order_magnitude"][-1]))
        backward_order_max.append(float(np.max(result["order_magnitude"])))

    return {
        "forward": {
            "A0": np.array(forward_A, dtype=np.float64),
            "order_end": np.array(forward_order_end, dtype=np.float64),
            "order_max": np.array(forward_order_max, dtype=np.float64),
        },
        "backward": {
            "A0": np.array(backward_A, dtype=np.float64),
            "order_end": np.array(backward_order_end, dtype=np.float64),
            "order_max": np.array(backward_order_max, dtype=np.float64),
        },
    }


def hysteresis_plot(
    sweep_result: Dict[str, Any],
    *,
    outfile: str,
    title: str = "Hysteresis sweep: |O| vs drive amplitude",
) -> None:
    import matplotlib.pyplot as plt

    f = sweep_result["forward"]
    b = sweep_result["backward"]

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(f["A0"], f["order_end"], marker="o", lw=1.8, label="forward (up)")
    plt.plot(b["A0"], b["order_end"], marker="s", lw=1.8, label="backward (down)")
    plt.xlabel("drive amplitude A0 (a.u.)")
    plt.ylabel("|O| at end")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

