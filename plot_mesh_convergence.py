"""
Plot mesh convergence results for the spherical geometry with B2p_tang_ipp.

Computes the L2 relative error of the FOM radiation factor against the
analytical solution for each mesh size (lc).

Usage:
    python plot_mesh_convergence.py              # default: error vs 1/lc
    python plot_mesh_convergence.py --lc         # error vs lc
    python plot_mesh_convergence.py --inv-lc     # error vs 1/lc
    python plot_mesh_convergence.py --ndofs      # error vs number of DOFs
    python plot_mesh_convergence.py --all        # all 3 plots side by side
    python plot_mesh_convergence.py --save fig_  # save to PDF
"""

import numpy as np
import json
import sys
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from operators_POO import compute_analytical_radiation_factor

DIM_P = 2
DIM_Q = 2
STR_OPE   = "b2p_tang"
RESULTS_DIR = Path(__file__).parent / "raw_results" / "CVmesh" / STR_OPE / f"{DIM_P}_{DIM_Q}"
RADIUS = 0.1


def load_all_results():
    """Load all mesh convergence JSON files and compute errors."""
    results = []
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)

        lc       = data['lc']
        ndofs    = data['ndofs']
        freqvec  = np.array(data['frequencies'])
        z_num    = np.array(data['Z_center']['real'])
        z_ana    = compute_analytical_radiation_factor(freqvec, RADIUS).real

        # L2 relative error
        error = np.sqrt(np.sum((z_num - z_ana)**2) / np.sum(z_ana**2))

        results.append({
            'lc':      lc,
            'ndofs':   ndofs,
            'error':   error,
            'freqvec': freqvec,
            'z_num':   z_num,
            'z_ana':   z_ana,
        })

    # Sort by lc descending (coarsest first)
    results.sort(key=lambda r: -r['lc'])
    return results


def add_slope_triangle(ax, x, y, order, x_pos_frac=0.5):
    """Add a convergence-rate triangle on log-log axes."""
    i = int(len(x) * x_pos_frac)
    if i >= len(x) - 1:
        i = len(x) - 2
    x0, x1 = x[i], x[i + 1]
    y0 = y[i]
    y1 = y0 * (x1 / x0) ** order

    ax.plot([x0, x1], [y0, y0], 'k-', linewidth=0.8)
    ax.plot([x1, x1], [y0, y1], 'k-', linewidth=0.8)
    ax.plot([x0, x1], [y0, y1], 'k--', linewidth=0.8)
    mid_x = np.sqrt(x0 * x1)
    ax.text(mid_x, y0 * 0.7, '1', ha='center', fontsize=9)
    ax.text(x1 * 1.1, np.sqrt(y0 * y1), str(order), ha='left', fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: error vs lc
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_vs_lc(results):
    fig, ax = plt.subplots(figsize=(8, 6))

    lcs    = [r['lc'] for r in results]
    errors = [r['error'] for r in results]

    ax.loglog(lcs, errors, 'o-', markersize=6, linewidth=2)

    ax.set_xlabel('Mesh size lc [m]', fontsize=14)
    ax.set_ylabel('Relative L2 error', fontsize=14)
    ax.set_title(f'Mesh convergence — Spherical, {STR_OPE}', fontsize=14)
    ax.grid(which='both', linestyle='--')
    ax.invert_xaxis()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: error vs 1/lc
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_vs_inv_lc(results):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sort by 1/lc ascending
    sorted_res = sorted(results, key=lambda r: 1.0 / r['lc'])
    inv_lcs = [1.0 / r['lc'] for r in sorted_res]
    errors  = [r['error'] for r in sorted_res]

    ax.loglog(inv_lcs, errors, 'o-', markersize=6, linewidth=2)
    #ax.plot(inv_lcs, errors, 'o-', markersize=6, linewidth=2)
    #ax.set_yscale('log')

    ax.set_xlabel('1 / lc [1/m]', fontsize=14)
    ax.set_ylabel('Relative L2 error', fontsize=14)
    ax.set_title(f'Mesh convergence — Spherical, {STR_OPE}', fontsize=14)
    ax.grid(which='both', linestyle='--')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: error vs number of DOFs
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_vs_ndofs(results):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sort by ndofs ascending
    sorted_res = sorted(results, key=lambda r: r['ndofs'])
    ndofs  = [r['ndofs'] for r in sorted_res]
    errors = [r['error'] for r in sorted_res]

    ax.loglog(ndofs, errors, 'o-', markersize=6, linewidth=2)

    ax.set_xlabel('Number of DOFs', fontsize=14)
    ax.set_ylabel('Relative L2 error', fontsize=14)
    ax.set_title(f'Mesh convergence — Spherical, {STR_OPE}', fontsize=14)
    ax.grid(which='both', linestyle='--')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: all 3 side by side
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Sort for each axis
    sorted_lc    = sorted(results, key=lambda r: r['lc'])
    sorted_inv   = sorted(results, key=lambda r: 1.0 / r['lc'])
    sorted_ndofs = sorted(results, key=lambda r: r['ndofs'])

    # Error vs lc
    ax1.loglog([r['lc'] for r in sorted_lc],
               [r['error'] for r in sorted_lc],
               'o-', markersize=6, linewidth=2)
    ax1.set_xlabel('lc [m]', fontsize=12)
    ax1.set_ylabel('Relative L2 error', fontsize=12)
    ax1.set_title('Error vs lc', fontsize=13)
    ax1.grid(which='both', linestyle='--')
    ax1.invert_xaxis()

    # Error vs 1/lc
    ax2.loglog([1.0 / r['lc'] for r in sorted_inv],
               [r['error'] for r in sorted_inv],
               'o-', markersize=6, linewidth=2, color='C1')
    ax2.set_xlabel('1/lc [1/m]', fontsize=12)
    ax2.set_ylabel('Relative L2 error', fontsize=12)
    ax2.set_title('Error vs 1/lc', fontsize=13)
    ax2.grid(which='both', linestyle='--')

    # Error vs ndofs
    ax3.loglog([r['ndofs'] for r in sorted_ndofs],
               [r['error'] for r in sorted_ndofs],
               'o-', markersize=6, linewidth=2, color='C2')
    ax3.set_xlabel('Number of DOFs', fontsize=12)
    ax3.set_ylabel('Relative L2 error', fontsize=12)
    ax3.set_title('Error vs DOFs', fontsize=13)
    ax3.grid(which='both', linestyle='--')

    fig.suptitle(f'Mesh convergence — Spherical, {STR_OPE}', fontsize=14)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mesh convergence results (spherical, B2p_tang_ipp)")
    parser.add_argument('--lc', action='store_true',
                        help='Plot error vs lc')
    parser.add_argument('--inv-lc', action='store_true',
                        help='Plot error vs 1/lc')
    parser.add_argument('--ndofs', action='store_true',
                        help='Plot error vs number of DOFs')
    parser.add_argument('--all', action='store_true',
                        help='Plot all 3 views side by side')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figures to this prefix (e.g. "meshcv_")')

    args = parser.parse_args()

    results = load_all_results()
    if not results:
        print(f"No results found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Loaded {len(results)} mesh convergence results:")
    for r in results:
        print(f"  lc={r['lc']:.1e}  ndofs={r['ndofs']:>7d}  error={r['error']:.4e}")

    # Default: show all if nothing specific requested
    show_default = not (args.lc or args.inv_lc or args.ndofs or args.all)

    figs = []

    if args.lc:
        figs.append(('error_vs_lc', plot_error_vs_lc(results)))
    if args.inv_lc or show_default:
        figs.append(('error_vs_inv_lc', plot_error_vs_inv_lc(results)))
    if args.ndofs:
        figs.append(('error_vs_ndofs', plot_error_vs_ndofs(results)))
    if args.all:
        figs.append(('error_all', plot_all(results)))

    if args.save:
        for name, fig in figs:
            fname = f"{args.save}{name}.pdf"
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Saved: {fname}")

    plt.show()
