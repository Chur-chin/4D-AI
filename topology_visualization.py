"""
Topology / diagnostics visualization for 4D phase fields.

Includes:
1) Order-parameter magnitude plots vs time
2) Plaquette winding-density maps on selected 2D slices of the 4D grid
3) Phase heatmaps on selected 2D slices
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _wrap_to_pi(dphi: np.ndarray) -> np.ndarray:
    """Wrap angle differences to (-π, π]."""
    twopi = 2.0 * np.pi
    return (dphi + np.pi) % twopi - np.pi


def order_magnitude_plot(
    times: np.ndarray,
    order_magnitude: np.ndarray,
    *,
    outfile: str,
    title: str = "Order magnitude |O(t)|",
) -> None:
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(times, order_magnitude, lw=1.8)
    plt.xlabel("time (a.u.)")
    plt.ylabel("|O(t)|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def phase_slice_heatmap(
    phi: np.ndarray,
    *,
    axes: Tuple[int, int] = (0, 1),
    fixed_indices: Tuple[int, int] = (0, 0),
    outfile: str,
    title: str = "Phase slice heatmap",
    cmap: str = "twilight",
) -> None:
    """
    Plot a 2D slice of the 4D phase field as heatmap.

    Parameters
    - axes: which two dimensions span the plotted plane
    - fixed_indices: indices for the remaining two dimensions
    """
    if phi.ndim != 4:
        raise ValueError("phi must have shape (L,L,L,L)")

    dims = [0, 1, 2, 3]
    a, b = axes
    fixed_dims = [d for d in dims if d not in (a, b)]
    if len(fixed_dims) != 2:
        raise ValueError("axes must be two distinct dimensions in (0,1,2,3)")
    za = fixed_indices[0]
    zb = fixed_indices[1]
    fixed_map = {fixed_dims[0]: za, fixed_dims[1]: zb}

    L = phi.shape[0]
    i = np.arange(L)
    j = np.arange(L)

    # Build slice by indexing fixed dims
    # Use advanced indexing via a comprehension to keep it simple.
    slice2d = np.empty((L, L), dtype=np.float64)
    for ii in i:
        for jj in j:
            idx = [0, 0, 0, 0]
            idx[a] = ii
            idx[b] = jj
            idx[fixed_dims[0]] = fixed_map[fixed_dims[0]]
            idx[fixed_dims[1]] = fixed_map[fixed_dims[1]]
            slice2d[ii, jj] = phi[tuple(idx)]

    plt.figure(figsize=(6.8, 5.0))
    im = plt.imshow(slice2d, origin="lower", cmap=cmap, vmin=0.0, vmax=2.0 * np.pi)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="phi (rad)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def plaquette_winding_density(
    phi: np.ndarray,
    *,
    axes: Tuple[int, int] = (0, 1),
    fixed_indices: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Compute integer plaquette winding on a 2D slice.

    For each plaquette in the chosen plane, compute the wrapped sum of phase
    differences around the loop and quantize it to nearest integer winding.
    """
    if phi.ndim != 4:
        raise ValueError("phi must have shape (L,L,L,L)")

    dims = [0, 1, 2, 3]
    a, b = axes
    fixed_dims = [d for d in dims if d not in (a, b)]
    za = fixed_indices[0]
    zb = fixed_indices[1]
    fixed_map = {fixed_dims[0]: za, fixed_dims[1]: zb}
    L = phi.shape[0]

    def at(ii: int, jj: int) -> float:
        idx = [0, 0, 0, 0]
        idx[a] = ii
        idx[b] = jj
        idx[fixed_dims[0]] = fixed_map[fixed_dims[0]]
        idx[fixed_dims[1]] = fixed_map[fixed_dims[1]]
        return float(phi[tuple(idx)])

    winding = np.zeros((L, L), dtype=np.int32)
    twopi = 2.0 * np.pi

    for i in range(L):
        ip = (i + 1) % L
        for j in range(L):
            jp = (j + 1) % L

            phi00 = at(i, j)
            phi10 = at(ip, j)
            phi01 = at(i, jp)
            phi11 = at(ip, jp)

            # Phase differences along edges with wrapping
            dphi_x = _wrap_to_pi(phi10 - phi00)
            dphi_y = _wrap_to_pi(phi01 - phi00)
            dphi_y_x = _wrap_to_pi(phi11 - phi10)
            dphi_x_y = _wrap_to_pi(phi11 - phi01)

            # Sum around loop (counter-clockwise)
            sum_dphi = dphi_x + dphi_y_x - dphi_x_y - dphi_y
            w = int(np.rint(sum_dphi / twopi))
            winding[i, j] = w

    return winding


def winding_density_plot(
    winding: np.ndarray,
    *,
    outfile: str,
    title: str = "Plaquette winding density",
) -> None:
    if winding.ndim != 2:
        raise ValueError("winding must be 2D on a slice")

    plt.figure(figsize=(6.6, 5.0))
    vmax = np.max(np.abs(winding)) + 1e-9
    im = plt.imshow(winding, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="winding (int)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def topology_summary_index(winding: np.ndarray) -> float:
    """A single scalar summary: mean absolute winding density."""
    return float(np.mean(np.abs(winding)))

