import numpy as np
from scipy import special
from geometries import spherical_domain, curvedcubic_domain, new_broken_cubic_domain_CAD_side_box_sweep

import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time
from operators_POO import (Mesh,
                           B1p, B2p_tang, B2p,
                           Loading,
                           Simulation,
                           save_json,
                           compute_analytical_radiation_factor)

print("Start of the main_ABC_sidebox_sweep.py script")

# Fixed parameters
geometry1 = 'new_broken_cubic'
geo_fct   = new_broken_cubic_domain_CAD_side_box_sweep
radius    = 0.1
rho0      = 1.21
c0        = 343.8
freq      = 1200
freqvec   = np.array([freq])

dimP    = 4
dimQ    = 4
str_ope = "b2p_tang"

# Side_box sweep parameters
side_box_min = 0.11
side_box_max = 0.40
side_box_step = 0.05
side_box_vec = np.arange(side_box_min, side_box_max + side_box_step/2, side_box_step)

# Linear interpolation for lc: lc(0.11)=1e-2, lc(0.40)=2.5e-2
lc_min = 1e-2
lc_max = 2.5e-2
lc_vec = lc_min + (lc_max - lc_min) / (side_box_max - side_box_min) * (side_box_vec - side_box_min)

# Analytical radiation factor at 1200 Hz (computed once)
Z_analytical = compute_analytical_radiation_factor(freqvec, radius)
sigma_ana = Z_analytical.real[0]
print(f"Analytical radiation factor at {freq} Hz: {sigma_ana}")


def main_sidebox_sweep(dimP, dimQ, str_ope, freqvec, geo_fct,
                       side_box_vec, lc_vec, sigma_ana,
                       save_data=True):

    Z_center_real_list = []

    for ii, (side_box, lc) in enumerate(zip(side_box_vec, lc_vec)):
        print(f"\n--- side_box = {side_box:.2f}, lc = {lc:.4f} ({ii+1}/{len(side_box_vec)}) ---")

        mesh_   = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)
        loading = Loading(mesh_)

        if str_ope == "b1p":
            ope = B1p(mesh_)
        elif str_ope == "b2p":
            ope = B2p(mesh_)
        elif str_ope == "b2p_tang":
            ope = B2p_tang(mesh_)
        else:
            print("Operator doesn't exist")
            return

        simu = Simulation(mesh_, ope, loading)
        X_sol_FOM = simu.FOM(freqvec)
        Z_center = simu.compute_radiation_factor(freqvec, X_sol_FOM)
        Z_center_real_list.append(Z_center.real[0])

        print(f"  Z_center.real = {Z_center.real[0]:.6f}, analytical = {sigma_ana:.6f}")

    Z_center_real_arr = np.array(Z_center_real_list)

    if save_data:
        if geo_fct == spherical_domain:
            geometry1 = 'spherical'
        elif geo_fct == curvedcubic_domain:
            geometry1 = 'curvedcubic'
        elif geo_fct == new_broken_cubic_domain_CAD_side_box_sweep:
            geometry1 = 'new_broken_cubic'
        data = {
            'geometry': geometry1,
            'frequency': freq,
            'ope': str_ope,
            'dimP': dimP,
            'dimQ': dimQ,
            'side_box_vec': side_box_vec.tolist(),
            'lc_vec': lc_vec.tolist(),
            'Z_center_real': Z_center_real_arr.tolist()
        }
        filename = f"FOM/sidebox_sweep_{geometry1}_{str_ope}_{dimP}_{dimQ}_{freq}Hz"
        save_json(data, filename)
        print(f"\nResults saved to raw_results/{filename}.json")

    return Z_center_real_arr


##############################################

if __name__ == "__main__":

    Z_center_real_arr = main_sidebox_sweep(
        dimP        = dimP,
        dimQ        = dimQ,
        str_ope     = str_ope,
        freqvec     = freqvec,
        geo_fct     = geo_fct,
        side_box_vec = side_box_vec,
        lc_vec      = lc_vec,
        sigma_ana   = sigma_ana,
        save_data   = True
    )

    # Error: |sigma_sim - sigma_ana|^2
    error = (Z_center_real_arr - sigma_ana)**2

    # Sort by 1/side_box (ascending)
    inv_side_box = 1.0 / side_box_vec
    sort_idx = np.argsort(inv_side_box)
    inv_side_box_sorted = inv_side_box[sort_idx]
    error_sorted = error[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.semilogy(inv_side_box_sorted, error_sorted, '-o', label=f'{geometry1}, P{dimP}Q{dimQ}, f={freq} Hz')
    
    ax.set_xlabel(r'$1 / L_{\mathrm{box}}$ (m$^{-1}$)')
    ax.set_ylabel(r'$|\sigma_{\mathrm{sim}} - \sigma_{\mathrm{ana}}|^2$')
    ax.set_title(f'Error vs inverse domain size at f = {freq} Hz')
    ax.grid(True, which='both')
    ax.legend()
    plt.tight_layout()
    plt.show()
