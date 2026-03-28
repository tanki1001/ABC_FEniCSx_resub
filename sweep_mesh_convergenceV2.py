"""
Adaptive mesh convergence study on B2p_tang_ipp.

Starting from a coarse mesh (lc_start), the script:
  1. Runs a FOM frequency sweep
  2. Computes the L2 relative error between the current and previous FOM
  3. If the error is above TOL, refines the mesh (lc *= REFINE_RATIO) and repeats
  4. Stops when the inter-mesh error drops below TOL or lc < lc_min

All intermediate results are saved to raw_results/CVmesh/ for plotting.

Usage:
    python sweep_mesh_convergenceV2.py                          # default settings
    python sweep_mesh_convergenceV2.py --lc-start 4e-2          # custom starting lc
    python sweep_mesh_convergenceV2.py --tol 1e-3               # custom tolerance
    python sweep_mesh_convergenceV2.py --ratio 0.7 --lc-min 3e-3
"""

import numpy as np
import json
import argparse
from pathlib import Path
from time import time
from geometries import (spherical_domain, curved_cubic_domain_CAD,
                        new_broken_cubic_domain_CAD)
from operators_POO import (Mesh, Loading, Simulation,
                           B2p_tang_ipp, B2p_tang)

# ── Configuration ──────────────────────────────────────────────────────────
STR_OPE   = "b2p_tang_ipp"
DIM_P     = 2
DIM_Q     = 2
RADIUS    = 0.1
FREQVEC   = np.arange(1500, 2201, 50)

RESULTS_DIR = Path(__file__).parent / "raw_results" / "CVmesh"

GEO_MAP = {
    'spherical':        spherical_domain,
    'curvedcubic':      curved_cubic_domain_CAD,
    'new_broken_cubic': new_broken_cubic_domain_CAD,
}

OPE_MAP = {
    'b2p_tang_ipp': B2p_tang_ipp,
    'b2p_tang':     B2p_tang,
}


def run_fom_at_lc(lc, geometry1, geometry2, side_box, geo_fct):
    """Build mesh + operator, run FOM, return radiation factor and metadata."""
    print(f"\n  Building mesh (lc={lc:.4e})...")
    t_mesh_start = time()
    mesh_   = Mesh(DIM_P, DIM_Q, side_box, RADIUS, lc, geo_fct)
    ope_cls = OPE_MAP[STR_OPE]
    ope     = ope_cls(mesh_)
    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)
    t_mesh  = time() - t_mesh_start

    P, Q    = mesh_.fonction_spaces()
    ndofs_P = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
    ndofs_Q = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    ndofs   = ndofs_P + ndofs_Q
    print(f"  DOFs: {ndofs} (P: {ndofs_P}, Q: {ndofs_Q})")

    print(f"  Running FOM sweep: {FREQVEC[0]}-{FREQVEC[-1]} Hz "
          f"({len(FREQVEC)} points)...")
    t_fom_start = time()
    X_sol_FOM = simu.FOM(FREQVEC)
    t_fom = time() - t_fom_start

    Z_center = simu.compute_radiation_factor(FREQVEC, X_sol_FOM)

    info = {
        'lc':      lc,
        'ndofs':   ndofs,
        'ndofs_P': ndofs_P,
        'ndofs_Q': ndofs_Q,
        't_mesh':  t_mesh,
        't_fom':   t_fom,
    }
    return Z_center, info


def save_result(Z_center, info, geometry1, geometry2, inter_error):
    """Save a single mesh convergence result to JSON."""
    lc = info['lc']
    data = {
        'geometry1':   geometry1,
        'geometry2':   geometry2,
        'ope':         STR_OPE,
        'lc':          lc,
        'dimP':        DIM_P,
        'dimQ':        DIM_Q,
        'ndofs':       info['ndofs'],
        'ndofs_P':     info['ndofs_P'],
        'ndofs_Q':     info['ndofs_Q'],
        'inter_mesh_error': inter_error,
        'frequencies': FREQVEC.tolist(),
        'Z_center': {
            'real': Z_center.real.tolist(),
            'imag': Z_center.imag.tolist(),
        },
        'CPU_time': {
            'mesh_setup': info['t_mesh'],
            'fom_sweep':  info['t_fom'],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = (RESULTS_DIR /
             f"{geometry1}_{geometry2}_{STR_OPE}_{lc}_{DIM_P}_{DIM_Q}.json")
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved: {fname.name}")


def adaptive_mesh_convergence(geometry1, geometry2, side_box,
                              lc_start, tol, refine_ratio, lc_min, max_iter):
    """Adaptive mesh refinement loop."""
    geo_fct = GEO_MAP[geometry1]

    print(f"\n{'='*60}")
    print(f"  Adaptive mesh convergence: {geometry1} ({geometry2})")
    print(f"  lc_start={lc_start}, tol={tol}, ratio={refine_ratio}, "
          f"lc_min={lc_min}")
    print(f"{'='*60}")

    history = []
    lc = lc_start
    Z_prev = None

    for iteration in range(max_iter):
        print(f"\n--- Iteration {iteration + 1}, lc = {lc:.4e} ---")

        Z_current, info = run_fom_at_lc(lc, geometry1, geometry2,
                                         side_box, geo_fct)

        # Compute inter-mesh L2 relative error
        if Z_prev is not None:
            inter_error = np.sqrt(
                np.sum(np.abs(Z_current.real - Z_prev.real)**2)
                / np.sum(np.abs(Z_prev.real)**2)
            )
        else:
            inter_error = float('inf')

        history.append({
            'lc':          lc,
            'ndofs':       info['ndofs'],
            'inter_error': inter_error,
            't_fom':       info['t_fom'],
        })

        save_result(Z_current, info, geometry1, geometry2, inter_error)

        print(f"  Inter-mesh L2 error: {inter_error:.4e}")
        print(f"  FOM time: {info['t_fom']:.1f}s, ndofs: {info['ndofs']}")

        # Check convergence
        if inter_error < tol:
            print(f"\n  CONVERGED at lc={lc:.4e} (error={inter_error:.4e} < "
                  f"tol={tol})")
            break

        # Prepare next iteration
        Z_prev = Z_current
        lc *= refine_ratio

        if lc < lc_min:
            print(f"\n  STOPPED: lc={lc:.4e} below lc_min={lc_min}")
            break
    else:
        print(f"\n  STOPPED: max iterations ({max_iter}) reached")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary: {geometry1}")
    print(f"{'='*60}")
    print(f"  {'Iter':>4}  {'lc':>10}  {'ndofs':>8}  {'error':>12}  {'CPU':>8}")
    print(f"  {'-'*50}")
    for i, h in enumerate(history):
        err_str = f"{h['inter_error']:.4e}" if h['inter_error'] != float('inf') else "     ---"
        print(f"  {i+1:>4}  {h['lc']:>10.4e}  {h['ndofs']:>8d}  "
              f"{err_str:>12}  {h['t_fom']:>7.1f}s")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive mesh convergence for B2p_tang_ipp")
    parser.add_argument('--geometry', type=str, default='spherical',
                        choices=list(GEO_MAP.keys()),
                        help='Geometry to use')
    parser.add_argument('--geometry2', type=str, default='small',
                        help='Geometry size (small/large)')
    parser.add_argument('--side-box', type=float, default=None,
                        help='Side box size (default: 0.11 for small, 0.4 for large)')
    parser.add_argument('--lc-start', type=float, default=4e-2,
                        help='Starting (coarsest) mesh size')
    parser.add_argument('--tol', type=float, default=5e-4,
                        help='Convergence tolerance on inter-mesh L2 error')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='Mesh refinement ratio (lc_new = lc * ratio)')
    parser.add_argument('--lc-min', type=float, default=2e-3,
                        help='Minimum lc (safety stop)')
    parser.add_argument('--max-iter', type=int, default=15,
                        help='Maximum number of refinement steps')

    args = parser.parse_args()

    if args.side_box is not None:
        side_box = args.side_box
    elif args.geometry2 == 'large':
        side_box = 0.4
    else:
        side_box = 0.11

    adaptive_mesh_convergence(
        geometry1    = args.geometry,
        geometry2    = args.geometry2,
        side_box     = side_box,
        lc_start     = args.lc_start,
        tol          = args.tol,
        refine_ratio = args.ratio,
        lc_min       = args.lc_min,
        max_iter     = args.max_iter,
    )
