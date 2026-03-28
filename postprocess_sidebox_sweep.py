import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import special

# Physical parameters
radius = 0.1
c0     = 343.8


def compute_analytical_radiation_factor(freqvec, radius):
    k_output = 2 * np.pi * freqvec / c0
    Z_analytical = (1 - 2*special.jv(1, 2*k_output*radius) / (2*k_output*radius)
                    + 1j * 2*special.struve(1, 2*k_output*radius) / (2*k_output*radius))
    return Z_analytical


def import_json(filename):
    with open(f"./raw_results/{filename}.json", 'r') as f:
        data = json.load(f)
    return data


def load_and_plot(filename, ax, sigma_ana):
    data = import_json(filename)

    geometry      = data['geometry']
    freq          = data['frequency']
    dimP          = data['dimP']
    dimQ          = data['dimQ']
    side_box_vec  = np.array(data['side_box_vec'])
    Z_center_real = np.array(data['Z_center_real'])

    error = (Z_center_real - sigma_ana)**2

    
    if sort_inv:
        inv_side_box = 1.0 / side_box_vec
        sort_idx = np.argsort(inv_side_box)

        ax.semilogy(inv_side_box[sort_idx], error[sort_idx], '-o',
                    label=f'{geometry}, P{dimP}Q{dimQ}, f={freq} Hz')
    else:
        #ax.semilogy(side_box_vec, error, '-o',
        #            label=f'{geometry}, P{dimP}Q{dimQ}, f={freq} Hz')
        ax.plot(side_box_vec, error, '-o',
                    label=f'{geometry}, P{dimP}Q{dimQ}, f={freq} Hz')


##############################################

if __name__ == "__main__":

    sort_inv = False

    freq = 1200
    freqvec = np.array([freq])
    Z_analytical = compute_analytical_radiation_factor(freqvec, radius)
    sigma_ana = Z_analytical.real[0]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Add all sweep files to plot here
    load_and_plot("FOM/sidebox_sweep_spherical_b2p_tang_4_4_1200Hz", ax, sigma_ana)
    load_and_plot("FOM/sidebox_sweep_curvedcubic_b2p_tang_4_4_1200Hz", ax, sigma_ana)
    load_and_plot("FOM/sidebox_sweep_new_broken_cubic_b2p_tang_4_4_1200Hz", ax, sigma_ana)
    
    if sort_inv:
        ax.set_xlabel(r'$1 / L_{\mathrm{box}}$ (m$^{-1}$)')
    else:
        ax.set_xlabel(r'$L_{\mathrm{box}}$ (m)')
    ax.set_ylabel(r'$|\sigma_{\mathrm{sim}} - \sigma_{\mathrm{ana}}|^2$')
    ax.set_title(f'Error vs inverse domain size at f = {freq} Hz')
    ax.grid(True, which='both')
    ax.legend()
    plt.tight_layout()
    plt.show()
