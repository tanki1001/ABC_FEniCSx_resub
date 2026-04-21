import numpy as np
from geometries import spherical_domain, curvedcubic_domain, new_broken_cubic_domain_CAD

import matplotlib.pyplot as plt
from time import time

from operators_POO import (Mesh, Loading, Simulation,
                            B2p_tang_ipp,
                            import_data, save_json,
                            complex_modal_basis)

print("Start of the main_modal.py script")
geometry1 = 'new_broken_cubic'
geometry2 = 'small'

if  geometry2 == 'small':
    side_box = 0.11
    lc       = 1.5e-2
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 1.5e-2
else:
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.11
    lc       = 8e-3

radius = 0.1
rho0   = 1.21
c0     = 343.8

if   geometry1 == 'spherical':
    geo_fct = spherical_domain
elif geometry1 == 'curvedcubic':
    geo_fct = curvedcubic_domain
elif geometry1 == 'new_broken_cubic':
    geo_fct = new_broken_cubic_domain_CAD
else:
    print("WARNING : May you choose an implemented geometry")


# Complex modal projection via the Quadratic Eigenvalue Problem (QEP)
#     [D1 + lambda D2 + lambda^2 D3] phi = 0,   lambda = jk
# With BSP=True  : V_tilde = [[Phi_p, 0], [0, Phi_q]], reduced size 2 * N_modes.
# With BSP=False : Vn = [phi_1, ..., phi_N_modes],      reduced size N_modes.
def fct_main_modal(
    degP,
    degQ,
    str_ope,
    freqvec,
    N_modes,
    f0,
    FOM_from_data,
    ax,
    BSP=True,
):
    print(lc)
    mesh_ = Mesh(degP, degQ, side_box, radius, lc, geo_fct)

    if str_ope == "b2p_tang_ipp":
        ope = B2p_tang_ipp(mesh_)
    else:
        print("complex_modal_basis currently only validated against b2p_tang_ipp")
        return

    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)

    if FOM_from_data:
        file_name = (geometry1 + '_' + geometry2 + '_' + str_ope + '_'
                     + str(lc) + '_' + str(degP) + '_' + str(degQ))
        freqvec, Z_center_real = import_data(file_name)
    else:
        X_sol_FOM     = simu.FOM(freqvec)
        Z_center_real = simu.compute_radiation_factor(freqvec, X_sol_FOM)

    t1 = time()

    Vn, CPU_eig, CPU_asm, CPU_split = complex_modal_basis(
        simu=simu, f0=f0, N_modes=N_modes, BSP=BSP)
    t2 = time()
    print(f'Complex modal basis CPU time : {t2 - t1:.2f} s'
          f'  (eig={CPU_eig:.2f} s, asm={CPU_asm:.4f} s, '
          f'split={CPU_split:.4f} s)')

    X_sol_MOR, solvingMOR = simu.moment_matching_MOR(Vn, freqvec)

    t3 = time()
    print(f'Whole CPU time : {t3 - t1:.2f} s')

    simu.plot_radiation_factor(ax, freqvec, Z_center_real,
                               s='FOM_' + str_ope, compute=False)
    simu.plot_radiation_factor(ax, freqvec, X_sol_MOR,
                               s='ComplexModal_' + str_ope)

    save_ROM_results = False
    if save_ROM_results:
        Z_center = simu.compute_radiation_factor(freqvec, X_sol_MOR)
        data = {
            'geometry1' : geometry1,
            'geometry2' : geometry2,
            'ope'       : str_ope,
            'lc'        : lc,
            'dimP'      : degP,
            'dimQ'      : degQ,
            'N_modes'   : N_modes,
            'f0'        : f0,
            'frequencies' : freqvec.tolist(),
            'Z_center'  : {
                'real' : Z_center.real.tolist(),
                'imag' : Z_center.imag.tolist(),
            },
            'CPU_time'  : {
                'eigenproblem'  : CPU_eig,
                'assembling_Vn' : CPU_asm,
                'solvingMOR'    : solvingMOR,
            },
        }
        print(data['CPU_time'])
        filename = ("MOR/COMPLEX_MODAL" + data['geometry1'] + "_"
                    + data['geometry2'] + "_" + data['ope'] + "_"
                    + str(data['lc']) + "_" + str(data['dimP']) + "_"
                    + str(data['dimQ']) + "_" + str(data['N_modes']) + "_"
                    + str(data['f0']) + "Hz")
        save_json(data, filename)


str_ope = "b2p_tang_ipp"
dimP    = 2
dimQ    = dimP
N_modes = 50
f0      = 2000
freqvec = np.arange(80, 2001, 20)


if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(16, 9))

    BSP = True

    fct_main_modal(
        degP          = dimP,
        degQ          = dimQ,
        str_ope       = str_ope,
        freqvec       = freqvec,
        N_modes       = N_modes,
        f0            = f0,
        FOM_from_data = True,
        ax            = ax,
        BSP           = BSP,
    )
    reduced_size = 2 * N_modes if BSP else N_modes
    plt.title(f'Complex-modal ROM (QEP): N_modes={N_modes}, BSP={BSP}, '
              f'reduced size = {reduced_size}')
    plt.show()
