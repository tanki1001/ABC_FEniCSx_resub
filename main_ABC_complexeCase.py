import numpy as np
from geometries import biSpherical_domain_CAD, cubicSpherical_domain_CAD

import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time
from operators_POO import (Mesh,
                           B1p, B2p_tang, B2p, B2p_tang_ipp,
                           Loading_monopole,
                           Simulation,
                           save_json, import_data,
                           plot_pressure_slice_yz)

print("Start of the main_ABC_complexeCase.py script")

# Choice of the geometry among provided ones
geometry1 = 'cubicSpherical'
geometry2 = 'small'
geometry  = geometry1 + '_' + geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 1e-2
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2.5e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.11
    lc       = 1e-2

if   geometry1 == 'biSpherical':
    geo_fct = biSpherical_domain_CAD
elif geometry1 == 'cubicSpherical':
    geo_fct = cubicSpherical_domain_CAD
else :
    print("WARNING : May you choose an implemented geometry")

# Simulation parameters
radius  = 0.1
rho0    = 1.21
c0      = 343.8
freqvec = np.arange(80, 2001, 20) 

# Monopole parameters
Q_monopole = 0.01
x0 = [0, 0.08, 0]


def import_p_values(filename, tree='FOM/'):
    '''Import saved p_values data (monopole case).'''
    from operators_POO import import_json
    data = import_json(f'{tree}{filename}')
    freqvec = np.array(data['frequencies'])
    p_values = np.array(data['p_values']['real']) + 1j * np.array(data['p_values']['imag'])
    return freqvec, p_values


def main_ABC_complexeCase_fct(dimP,
                               dimQ,
                               str_ope,
                               freqvec,
                               geo_fct,
                               ax,
                               from_data=False,
                               save_data=False,
                               plot_pressure_field=False):

    s1 = geometry + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)

    if from_data:
        filename = geometry1 + "_" + geometry2 + "_" + str_ope + "_" + str(lc) + "_" + str(dimP) + "_" + str(dimQ)
        freqvec, p_values = import_p_values(filename)
    else:
        mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)
        loading  = Loading_monopole(mesh_, Q=Q_monopole)

        if str_ope == "b1p":
            ope = B1p(mesh_)
        elif str_ope == "b2p":
            ope = B2p(mesh_)
        elif str_ope == "b2p_tang":
            ope = B2p_tang(mesh_)
        elif str_ope == "b2p_tang_ipp":
            ope = B2p_tang_ipp(mesh_)
        else:
            print("Operator doesn't exist")
            return

        simu = Simulation(mesh_, ope, loading)

        X_sol_FOM = simu.FOM(freqvec)
        p_values = simu.evaluate_at_point(X_sol_FOM, freqvec, x0)

        if save_data:
            data = {
                'geometry1': geometry1,
                'geometry2': geometry2,
                'ope': str_ope,
                'lc': lc,
                'dimP': dimP,
                'dimQ': dimQ,
                'x0': x0,
                'Q_monopole': Q_monopole,
                'frequencies': freqvec.tolist(),
                'p_values': {
                    'real': p_values.real.tolist(),
                    'imag': p_values.imag.tolist()
                }}
            filename = "FOM/" + data['geometry1'] + "_" + data['geometry2'] + "_" + data['ope'] + "_" + str(data['lc']) + "_" + str(data['dimP']) + "_" + str(data['dimQ'])
            save_json(data, filename)

        if plot_pressure_field:
            freq_plot = freqvec[-1]
            idx_plot  = np.argmin(np.abs(freqvec - freq_plot))
            P, _ = mesh_.fonction_spaces()
            plot_pressure_slice_yz(P, X_sol_FOM[idx_plot],
                                title=f'|p| at f = {freqvec[idx_plot]} Hz, (y,z) plane')

    print(f'Pressure amplitude at x0={x0} for {s1}: {p_values}')
    #ax.plot(freqvec, np.abs(p_values), label=s1)
    ax.plot(freqvec, np.real(p_values), label=s1)
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$|p(x_0)|$')
    ax.set_title(f'Pressure amplitude at x0 = {x0}')


def load_comsol_data(filepath):
    '''Load COMSOL point graph export (freq, pressure) skipping header lines starting with %.'''
    freq_list, p_list = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            freq_list.append(float(parts[0]))
            p_list.append(float(parts[1]))
    return np.array(freq_list), np.array(p_list)


##############################################

if __name__ == "__main__":

    dimP = 2
    dimQ = 2
    str_ope = "b2p_tang_ipp"

    fig, ax = plt.subplots(figsize=(16, 9))

    # COMSOL reference (PML)
    comsol_freq, comsol_p = load_comsol_data('COMSOL_BEM.txt')
    ax.plot(comsol_freq, comsol_p, '--k', label='COMSOL BEM')

    main_ABC_complexeCase_fct(dimP     = dimP,
                               dimQ    = dimQ,
                               str_ope = str_ope,
                               freqvec = freqvec,
                               geo_fct = geo_fct,
                               ax      = ax,
                               from_data           = False,
                               save_data           = False,
                               plot_pressure_field = False)
    

    plt.savefig("claude_plot.png")
    plt.tight_layout()
    plt.show()
