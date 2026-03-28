import numpy as np
from scipy import special
from geometries import spherical_domain, curvedcubic_domain, new_broken_cubic_domain_CAD

import petsc4py

import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from operators_POO import (Mesh, Loading, Simulation,
                            B1p, B2p,
                            B2p_tang,
                            import_data, save_json)

print("Start of the main_wcawe.py script")
geometry1 = 'new_broken_cubic'
geometry2 = 'small'

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 1e-2
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2.5e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.11
    lc       = 8e-3

radius   = 0.1
rho0 = 1.21
c0   = 343.8

if   geometry1 == 'spherical':
    geo_fct = spherical_domain
elif geometry1 == 'curvedcubic':
    geo_fct = curvedcubic_domain
elif geometry1 == 'new_broken_cubic':
    geo_fct = new_broken_cubic_domain_CAD

else :
    print("WARNING : May you choose an implemented geometry")

def fct_main_wcawe(
    degP,
    degQ,
    str_ope,
    freqvec,
    list_N,
    list_freq,
    BSP,
    FOM_from_data,
    ax
):


    
    mesh_    = Mesh(degP, degQ, side_box, radius, lc, geo_fct)

    if str_ope == "b1p":
        ope = B1p(mesh_)
    elif str_ope == "b2p":
        ope = B2p(mesh_)
    elif str_ope == "b2p_tang":
        ope = B2p_tang(mesh_)
    else:
        print("Operator doesn't exist")
        return

    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)
    if FOM_from_data :
        file_name = geometry1 + '_' + geometry2 + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)
        freqvec, Z_center_real = import_data(file_name)
    else:
        PavFOM_fct = simu.FOM(freqvec)
        Z_center_real = simu.compute_radiation_factor(freqvec, PavFOM_fct)


    t1   = time()
    Vn, CPUbuildingVn, CPUderivatives, CPUspliting_Vn = simu.merged_WCAWE(list_N, list_freq, BSP = BSP)
    t2   = time()
    print(f'WCAWE CPU time  : {t2 - t1}')

    
    t3 = time()
    #freqvecMOR = np.arange(80, 2001, 1)
    PavWCAWE_fct, solvingMOR = simu.moment_matching_MOR(Vn, freqvec)

    t4 = time()
    print(f'Whole CPU time  : {t4 - t1}')

    simu.plot_radiation_factor(ax, freqvec, Z_center_real, s = 'FOM_' + str_ope, compute = False)
    simu.plot_radiation_factor(ax, freqvec, PavWCAWE_fct, s = 'WCAWE_'+ str_ope)
    save_ROM_results = True
    if save_ROM_results:
        Z_center = simu.compute_radiation_factor(freqvec, PavWCAWE_fct)
        data = {
            'geometry1': geometry1,
            'geometry2': geometry2,
            'ope': str_ope,
            'lc': lc,
            'dimP': dimP,
            'dimQ': dimQ,
            'N' : list_N[0],
            'f0' : list_freq[0],
            'frequencies': freqvec.tolist(),
            'Z_center': {
                'real': Z_center.real.tolist(),
                'imag': Z_center.imag.tolist()
            },
            'CPU_time' : {
                'derivatives' : CPUderivatives,
                'buildingVn'  : CPUbuildingVn,
                'assembling_C': 0,
                'spliting_Vn' : CPUspliting_Vn,
                'solvingMOR'  : solvingMOR
            }}
        print(data['CPU_time'])
        print(f'Sum up on time : {data['CPU_time']['derivatives'] + data['CPU_time']['buildingVn'] + data['CPU_time']['spliting_Vn'] + data['CPU_time']['solvingMOR'] }')
        
        filename = "MOR/" + data['geometry1'] + "_" + data['geometry2'] + "_" + data['ope'] + "_" + str(data['lc']) + "_" + str(data['dimP']) + "_" + str(data['dimQ']) + "_" + str(data['N'])+ "_" + str(data['f0']) + "Hz"
        
        save_json(data, filename)


str_ope = "b2p_tang"
dimP = 4
dimQ = dimP
list_N = [10]
list_freq = [1000]
freqvec = np.arange(80, 2001, 20)


if True :
    for N in [10]:
        list_N = [N]
        fig, ax = plt.subplots(figsize=(16,9))

        fct_main_wcawe(
            degP      = dimP,
            degQ      = dimP,
            str_ope   = str_ope,
            freqvec   = freqvec,
            list_N    = list_N,
            list_freq = list_freq,
            BSP           = True,
            FOM_from_data = True,
            ax        = ax,
        )
    if False:
        fct_main_wcawe(
            degP      = dimP,
            degQ      = dimP,
            str_ope   = "b2p_tang",
            freqvec   = freqvec,
            list_N    = list_N,
            list_freq = list_freq,
            BSP       = True,
            FOM_from_data = True,
            ax        = ax,
        )
    plt.title(f'ROM with N = {list_N}')
    plt.show()


