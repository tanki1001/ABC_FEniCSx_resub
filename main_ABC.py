# Modules importations
import numpy as np
from scipy import special
from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain, ellipsoidal_domain, curvedcubic_domain, half_curvedcubic_domain, new_broken_cubic_domain, new_broken_cubic_domain_CAD, curved_cubic_domain_CAD

from dolfinx.fem import (form, Function, FunctionSpace, petsc)
import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from operators_POO import (Mesh,
                           B1p, B2p_tang, B2p, B2p_tang_ipp,
                        Loading, 
                        Simulation, 
                        plot_analytical_result_sigma, save_json, import_data)

print("Start of the main_ABC.py script")

# Choice of the geometry among provided ones
geometry1 = 'new_broken_cubic'  # 'spherical', 'curvedcubic', 'new_broken_cubic'
geometry2 = 'large'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 2e-2
    #lc       = 1e-2
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.40
    lc       = 1e-2 #Typical mesh size : Small case : 8e-3 Large case : 2e-3

if   geometry1 == 'cubic':
    geo_fct = cubic_domain
elif geometry1 == 'spherical':
    geo_fct = spherical_domain
elif geometry1 == 'half_cubic':
    geo_fct = half_cubic_domain
elif geometry1 == 'broken_cubic':
    geo_fct = broken_cubic_domain
elif geometry1 == 'ellipsoidal':
    geo_fct = ellipsoidal_domain
elif geometry1 == 'curvedcubic':
    #geo_fct = curvedcubic_domain
    geo_fct = curved_cubic_domain_CAD
elif geometry1 == 'half_curvedcubic':
    geo_fct = half_curvedcubic_domain
elif geometry1 == 'new_broken_cubic':
    #geo_fct = new_broken_cubic_domain
    geo_fct = new_broken_cubic_domain_CAD
else :
    print("WARNING : May you choose an implemented geometry")

# Simulation parameters
radius  = 0.1                               # Radius of the baffle
rho0    = 1.21                              # Density of the air
c0      = 343.8                             # Speed of sound in air
freqvec = np.arange(80, 2001, 20)        # List of the frequencies



def main_ABCmodified_curves_fct(dimP,
                                dimQ,
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                save_data           = False,
                                plot_row_columns    = False,
                                plot_heatmap        = False,
                                plot_cond           = False,
                                plot_svlistZ        = False,
                                plot_pressure_field = False):
    
    mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)
    loading  = Loading(mesh_)

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

    simu    = Simulation(mesh_, ope, loading)

    if "modified" in str_ope:
        s1 = 'modified_ope/'
    else:
        s1 = 'classical/classical_'
    
    s2 = geometry + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)
    s1 += s2
    
    if from_data:
        filename = geometry1 + "_" + geometry2 + "_" + str_ope + "_" + str(lc) + "_" + str(dimP) + "_" + str(dimQ)
        freqvec, Z_center_data = import_data(filename)
    else :
        freqvec = freqvec
        X_sol_FOM = simu.FOM(freqvec)
        if save_data:
            Z_center = simu.compute_radiation_factor(freqvec, X_sol_FOM)
            if from_data == True:
                print("Data already exists, not saving.")
            else:
                data = {
                    'geometry1': geometry1,
                    'geometry2': geometry2,
                    'ope': str_ope,
                    'lc': lc,
                    'dimP': dimP,
                    'dimQ': dimQ,
                    'computed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'frequencies': freqvec.tolist(),
                    'Z_center': {
                        'real': Z_center.real.tolist(),
                        'imag': Z_center.imag.tolist()
                    }}
                #filename = "FOM/Npi24/24"+data['geometry1'] + "_" + data['geometry2'] + "_" + data['ope'] + "_" + str(data['lc']) + "_" + str(data['dimP']) + "_" + str(data['dimQ'])
                filename = "FOM/"+data['geometry1'] + "_" + data['geometry2'] + "_" + data['ope'] + "_" + str(data['lc']) + "_" + str(data['dimP']) + "_" + str(data['dimQ'])
                save_json(data, filename)

    if from_data:
        simu.plot_radiation_factor(ax, freqvec, Z_center_data, s = s1, compute = False)
    else:
        simu.plot_radiation_factor(ax, freqvec, X_sol_FOM, s = s1, compute = True)

    if plot_row_columns:
        simu.plot_row_columns_norm(freq = 1750, s = s2)
    
    if plot_heatmap:
        simu.plot_matrix_heatmap(freq = 1750, s = s2)

    if plot_cond:
        simu.plot_cond(freqvec, s = s2)
    
    if plot_svlistZ:
        simu.plot_sv_listZ(s = s2)

    if plot_pressure_field :
        simu.singular_frequency_FOM(2000)



##############################################

if True:
    fig, ax = plt.subplots(figsize=(16,9))


    plot_analytical_result_sigma(ax, freqvec, radius)

    dimP = 2
    dimQ = 2
    str_ope = "b2p_tang_ipp"

    main_ABCmodified_curves_fct(dimP                = dimP,
                                dimQ                = dimQ, 
                                str_ope             = str_ope, 
                                from_data           = False, 
                                freqvec             = freqvec, 
                                geo_fct             = geo_fct, 
                                ax                  = ax, 
                                save_data           = True, 
                                plot_heatmap        = False, 
                                plot_pressure_field = False)
    if False:
        dimP = 4
        dimQ = 4
        str_ope = "b2p_tang"
        lc = 1e-2

        main_ABCmodified_curves_fct(dimP                = dimP,
                                    dimQ                = dimQ, 
                                    str_ope             = str_ope, 
                                    from_data           = False, 
                                    freqvec             = freqvec, 
                                    geo_fct             = geo_fct, 
                                    ax                  = ax, 
                                    save_data           = True, 
                                    plot_heatmap        = False, 
                                    plot_pressure_field = False)
    
    
    ax.legend()
    ax.set_ylim(0,1.25)
    plt.show()

    