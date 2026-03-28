"""
Sweep SOAR method on B2p_tang_ipp operator for N = 1..100
across spherical, curvedcubic, and new_broken_cubic geometries.

Usage:
    python sweep_soar_b2p_tang_ipp.py                  # run all 3 geometries
    python sweep_soar_b2p_tang_ipp.py spherical         # run only spherical
    python sweep_soar_b2p_tang_ipp.py curvedcubic       # run only curvedcubic
    python sweep_soar_b2p_tang_ipp.py new_broken_cubic  # run only new_broken_cubic
"""

import numpy as np
import sys
from time import time
from geometries import spherical_domain, curved_cubic_domain_CAD, new_broken_cubic_domain_CAD
from operators_POO import (Mesh, Loading, Simulation,
                           B2p_tang_ipp, import_data, save_json, soar)

# ── Configuration ──────────────────────────────────────────────────────────
str_ope   = "b2p_tang_ipp"
dimP      = 2
dimQ      = 2
f0        = 1000
freqvec   = np.arange(80, 2001, 20)
N_values  = list(range(1, 101))   # N from 1 to 100

# Geometry configs
GEOMETRY_CONFIGS = {
    'spherical': {
        'geometry2': 'large',
        'geo_fct':   spherical_domain,
        'side_box':  0.4,
        'lc':        1.5e-2,
        'radius':    0.1,
    },
    'curvedcubic': {
        'geometry2': 'large',
        'geo_fct':   curved_cubic_domain_CAD,
        'side_box':  0.4,
        'lc':        1.5e-2,
        'radius':    0.1,
    },
    'new_broken_cubic': {
        'geometry2': 'large',
        'geo_fct':   new_broken_cubic_domain_CAD,
        'side_box':  0.4,
        'lc':        1.5e-2,
        'radius':    0.1,
    },
}


def run_sweep(geometry1, config):
    """Run SOAR for N=1..100 on a single geometry."""
    geometry2 = config['geometry2']
    lc        = config['lc']

    print(f"\n{'='*60}")
    print(f"  Geometry: {geometry1} ({geometry2}), lc={lc}")
    print(f"{'='*60}")

    # ── Load FOM reference ──
    fom_filename = f"{geometry1}_{geometry2}_{str_ope}_{lc}_{dimP}_{dimQ}"
    print(f"Loading FOM data: {fom_filename}")
    freqvec_fom, Z_center_FOM = import_data(fom_filename)

    # ── Build mesh and operator (once per geometry) ──
    print("Building mesh and operator...")
    mesh_ = Mesh(dimP, dimQ, config['side_box'], config['radius'],
                 lc, config['geo_fct'])
    ope     = B2p_tang_ipp(mesh_)
    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)

    # ── Sweep over N ──
    for N in N_values:
        print(f"\n--- N = {N} ---")
        t_start = time()

        Vn, CPUbuildingVn, CPUderivatives, CPUspliting_Vn = soar(
            simu=simu, f0=f0, n=N
        )

        X_sol_MOR, solvingMOR = simu.moment_matching_MOR(Vn, freqvec)
        t_total = time() - t_start

        Z_center_ROM = simu.compute_radiation_factor(freqvec, X_sol_MOR)

        # ── Save results ──
        data = {
            'geometry1':   geometry1,
            'geometry2':   geometry2,
            'ope':         str_ope,
            'lc':          lc,
            'dimP':        dimP,
            'dimQ':        dimQ,
            'N':           N,
            'f0':          f0,
            'frequencies': freqvec.tolist(),
            'Z_center': {
                'real': Z_center_ROM.real.tolist(),
                'imag': Z_center_ROM.imag.tolist(),
            },
            'CPU_time': {
                'derivatives':  CPUderivatives,
                'buildingVn':   CPUbuildingVn,
                'spliting_Vn':  CPUspliting_Vn,
                'solvingMOR':   solvingMOR,
                'total':        t_total,
            },
        }

        filename = (f"MOR/SOAR/{geometry1}_{geometry2}_{str_ope}"
                    f"_{lc}_{dimP}_{dimQ}_{N}_{f0}Hz")
        save_json(data, filename)
        print(f"  Saved: {filename}  (total CPU: {t_total:.2f}s)")


if __name__ == "__main__":
    # Parse CLI args to optionally select a single geometry
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        selected = list(GEOMETRY_CONFIGS.keys())

    for geo_name in selected:
        if geo_name not in GEOMETRY_CONFIGS:
            print(f"Unknown geometry: {geo_name}. "
                  f"Choose from: {list(GEOMETRY_CONFIGS.keys())}")
            continue
        run_sweep(geo_name, GEOMETRY_CONFIGS[geo_name])

    print("\nAll sweeps done. Use plot_sweep_soar.py to visualize results.")
