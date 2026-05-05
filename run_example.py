"""
Run a small demonstration of the 4D global phase-field dynamics scaffold.

It will generate:
  - order magnitude plot
  - phase slice heatmap
  - winding density map
  - hysteresis sweep plot
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np

from phase_collective_sim import PhaseSimConfig, run_collective_dynamics
from material_calibration import GrapheneHBNInputs, calibrate_graphene_hbn
from topology_visualization import (
    order_magnitude_plot,
    phase_slice_heatmap,
    plaquette_winding_density,
    winding_density_plot,
)
from hysteresis_sweep import HysteresisSweepConfig, hysteresis_plot, hysteresis_sweep


def _parse_axes(s: str) -> Tuple[int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("axes must be like '0,1'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--twist_deg", type=float, default=1.5)
    ap.add_argument("--eps_hbn", type=float, default=4.5)
    ap.add_argument("--eps_env", type=float, default=2.5)

    ap.add_argument("--axes", type=str, default="0,1", help="plane axes for 2D slice, like '0,1'")
    ap.add_argument("--fixed_indices", type=str, default="4,4", help="fixed indices for remaining dims, like '2,3'")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    axes = _parse_axes(args.axes)
    fixed_parts = [int(x.strip()) for x in args.fixed_indices.split(",")]
    if len(fixed_parts) != 2:
        raise ValueError("fixed_indices must have two integers, e.g. '2,3'")

    # 1) Calibration -> overrides
    calib_in = GrapheneHBNInputs(
        twist_angle_deg=args.twist_deg,
        eps_hbn=args.eps_hbn,
        eps_env=args.eps_env,
        temperature_K=4.0,
        scale=1.0,
    )
    calib = calibrate_graphene_hbn(calib_in)

    # 2) Create simulation config
    cfg = PhaseSimConfig(
        L=int(args.L),
        dt=float(args.dt),
        steps=int(args.steps),
        seed=int(args.seed),
        coupling_global=calib["coupling_global"],
        laplacian_strength=calib["laplacian_strength"],
        potential_strength=calib["potential_strength"],
        damping=calib["damping"],
        hbar=calib.get("hbar", 1.0),
        drive_sigma=calib["drive_sigma"],
        # kick amplitude/omega are left at config defaults; we want to show qualitative behavior
        save_every=5,
    )

    # 3) Run simulation
    result = run_collective_dynamics(cfg, record_phi=False)
    times = result["times"]
    order_mag = result["order_magnitude"]
    phi_final = result["final_state"].phi

    order_plot_path = str(outdir / "order_magnitude.png")
    order_magnitude_plot(times, order_mag, outfile=order_plot_path)

    # 4) Topology diagnostics from final field
    phase_path = str(outdir / "phase_slice.png")
    phase_slice_heatmap(
        phi_final,
        axes=axes,
        fixed_indices=(fixed_parts[0], fixed_parts[1]),
        outfile=phase_path,
        title="Phase slice (final state)",
    )

    winding = plaquette_winding_density(phi_final, axes=axes, fixed_indices=(fixed_parts[0], fixed_parts[1]))
    winding_path = str(outdir / "winding_density.png")
    winding_density_plot(winding, outfile=winding_path, title="Plaquette winding density (final state)")

    # 5) Hysteresis sweep
    A_values = np.linspace(0.05, 0.40, 7)
    sweep_cfg = HysteresisSweepConfig(A_values=A_values, steps_per_point=max(80, args.steps // 5), save_every=10)
    sweep_result = hysteresis_sweep(cfg, sweep_cfg, seed_initial_state=args.seed)
    hysteresis_plot(
        sweep_result,
        outfile=str(outdir / "hysteresis.png"),
    )

    print("Saved outputs to:", outdir.resolve())


if __name__ == "__main__":
    main()

