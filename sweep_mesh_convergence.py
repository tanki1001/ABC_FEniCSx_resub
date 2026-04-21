"""
Mesh convergence study on the spherical geometry with B2p_tang_ipp.

For each lc value, runs a FOM frequency sweep on the high-frequency range
and saves the radiation factor + metadata (DOFs, CPU time) to JSON.

Usage:
    python sweep_mesh_convergence.py              # run all lc values
    python sweep_mesh_convergence.py 1.5e-2 1e-2  # run specific lc values
"""

import numpy as np
import sys
import json
from pathlib import Path
from time import time
from geometries import spherical_domain, new_broken_cubic_domain_CAD
from operators_POO import (Mesh, Loading, Simulation,
                           B2p_tang_ipp, B2p_tang, save_json)

# ── Configuration ──────────────────────────────────────────────────────────
STR_OPE   = "b2p_tang_ipp"
OPE_MAP = {
    'b2p_tang_ipp': B2p_tang_ipp,
    'b2p_tang':     B2p_tang,
}
DIM_P     = 4
DIM_Q     = 4
GEOMETRY1 = "spherical"
GEOMETRY2 = "small"
SIDE_BOX  = 0.11
RADIUS    = 0.1
FREQVEC   = np.arange(1500, 2201, 50)

LC_VALUES = [4e-2, 3e-2, 2e-2, 1.5e-2, 1e-2, 8e-3]
#LC_VALUES = np.arange(3e-3, 4e-2, 0.004)
#LC_VALUES = [1.5e-2, 1.3e-2, 1.1e-2, 1e-2, 8e-3, 6e-3]

RESULTS_DIR = Path(__file__).parent / "raw_results" / "CVmesh" / STR_OPE / f"{DIM_P}_{DIM_Q}"


def run_single_lc(lc):
    """Run FOM frequency sweep for a single lc value."""
    print(f"\n{'='*60}")
    print(f"  lc = {lc}, geometry = {GEOMETRY1} ({GEOMETRY2})")
    print(f"{'='*60}")

    # ── Build mesh and operator ──
    print("Building mesh and operator...")
    t_mesh_start = time()
    mesh_   = Mesh(DIM_P, DIM_Q, SIDE_BOX, RADIUS, lc, spherical_domain)
    ope_cls = OPE_MAP[STR_OPE]
    ope     = ope_cls(mesh_)
    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)
    t_mesh = time() - t_mesh_start

    # ── Get DOF count ──
    P, Q = mesh_.fonction_spaces()
    ndofs_P = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
    ndofs_Q = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    ndofs   = ndofs_P + ndofs_Q
    print(f"  DOFs: {ndofs} (P: {ndofs_P}, Q: {ndofs_Q})")

    # ── FOM frequency sweep ──
    print(f"  Running FOM sweep: {FREQVEC[0]}-{FREQVEC[-1]} Hz "
          f"({len(FREQVEC)} points)...")
    t_fom_start = time()
    X_sol_FOM = simu.FOM(FREQVEC)
    t_fom = time() - t_fom_start

    Z_center = simu.compute_radiation_factor(FREQVEC, X_sol_FOM)

    # ── Save results ──
    data = {
        'geometry1':   GEOMETRY1,
        'geometry2':   GEOMETRY2,
        'ope':         STR_OPE,
        'lc':          lc,
        'dimP':        DIM_P,
        'dimQ':        DIM_Q,
        'ndofs':       ndofs,
        'ndofs_P':     ndofs_P,
        'ndofs_Q':     ndofs_Q,
        'frequencies': FREQVEC.tolist(),
        'Z_center': {
            'real': Z_center.real.tolist(),
            'imag': Z_center.imag.tolist(),
        },
        'CPU_time': {
            'mesh_setup': t_mesh,
            'fom_sweep':  t_fom,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = RESULTS_DIR / f"{GEOMETRY1}_{GEOMETRY2}_{STR_OPE}_{lc}_{DIM_P}_{DIM_Q}.json"
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved: {fname.name}  (FOM: {t_fom:.1f}s, ndofs: {ndofs})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        selected = [float(v) for v in sys.argv[1:]]
    else:
        selected = LC_VALUES

    for lc in selected:
        run_single_lc(lc)

    print("\nAll mesh convergence sweeps done. "
          "Use plot_mesh_convergence.py to visualize results.")
