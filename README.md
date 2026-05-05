# 4D-AI (Global 4D Phase-Field Dynamics Scaffold)

This repository provides a small, runnable **4D phase-field simulation scaffold** based on two core ideas:

1. **Global phase-field coupling** (Kuramoto-like order parameter)
   - Order parameter: `O = <exp(i*phi)>` (spatial average over the 4D grid)
   - Each site is driven toward alignment with the global `O`.

2. **Momentum transfer built on** `p = ħ k`
   - A forcing term is applied in **Fourier space** with magnitude proportional to `|k|`,
     so the injected momentum scale follows `p = ħ k`.

The code is intentionally lightweight and designed for qualitative exploration and
topology diagnostics (winding on 2D slices).

## Files

- `phase_collective_sim.py`: main global 4D phase-field engine (`run_collective_dynamics`)
- `material_calibration.py`: graphene–hBN-inspired parameter mapping into simulation constants
- `topology_visualization.py`: order plots + phase slice heatmaps + plaquette winding density
- `hysteresis_sweep.py`: forward/back hysteresis sweep of drive amplitude
- `run_example.py`: runs the full demo and saves figures into `outputs/`

## Requirements

Tested with Python 3.10+.

Install dependencies:
```bash
pip install numpy matplotlib
```

## Quick Start

Run the demo:
```bash
python run_example.py
```

Outputs are written to `outputs/`:
- `order_magnitude.png`
- `phase_slice.png`
- `winding_density.png`
- `hysteresis.png`

## Customization

Key runtime parameters can be adjusted via CLI:

```bash
python run_example.py --L 8 --steps 500 --dt 0.05 --twist_deg 1.5 --eps_hbn 4.5 --eps_env 2.5
```

- `--axes` selects the 2D slice plane for visualization (default `0,1`)
- `--fixed_indices` selects indices for the other two dimensions (default `4,4`)

## Notes

This project is a scaffold: the calibration mapping and the equation-of-motion terms are
chosen to keep the simulation stable and nontrivial for experimentation, not to exactly
match a specific physical experiment.

