import os
import numpy as np
from scipy import special
import scipy.linalg as la
from scipy.sparse import csr_matrix 
from scipy.io import savemat
from sympy import symbols, diff, lambdify
import sympy as sy
import matplotlib.pyplot as plt

import pyvista

from tqdm import tqdm
from time import time

from abc import ABC, abstractmethod

import gmsh
from dolfinx import plot
from basix.ufl import element
from dolfinx.io import gmshio
import dolfinx.mesh as msh
from mpi4py import MPI
from dolfinx.fem import Function, functionspace, assemble, form, petsc, Constant, assemble_scalar
from dolfinx.fem.petsc import LinearProblem
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, inner, Measure, variable, FacetNormal, CellNormal)
import ufl
import petsc4py
from petsc4py import PETSc
import slepc4py.SLEPc as SLEPc

from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain, curvedcubic_domain
from operators_POO import ( Mesh,
                            plot_analytical_result_sigma, tangential_proj,compute_analytical_radiation_factor)

################
def save_json(data, filename):
    import json
    with open(f"raw_results/{filename}.json", 'w') as f:
        json.dump(data, f, indent=4)
################

# Choice of the geometry among provided ones
geometry1 = 'spherical'
geometry2 = 'small'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 2*8e-3
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
elif geometry1 == 'curvedcubic':
    geo_fct = curvedcubic_domain
elif geometry1 == 'half_cubic':
    geo_fct = half_cubic_domain
elif geometry1 == 'broken_cubic':
    geo_fct = broken_cubic_domain
else :
    print("WARNING : May you choose an implemented geometry")



# Simulation parameters
radius  = 0.1                               # Radius of the baffle
rho0    = 1.21                              # Density of the air
c0      = 343.8                             # Speed of sound in air
freqvec = np.arange(80, 2001, 00)        # List of the frequencies
#lc_list = np.arange(5e-3, 5e-2, 1e-3)
#lc_list = [5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 1.1e-2, 1.2e-2, 1.3e-2, 1.4e-2, 1.5e-2, 1.6e-2, 1.7e-2, 1.8e-2, 2e-2]
#lc_list = [8e-3, 1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2e-2]
#lc_list = [2e-2]
lc_list = [2e-2, 2.1e-2, 2.2e-2, 2.3e-2, 2.4e-2, 2.5e-2, 2.6e-2, 2.7e-2, 2.8e-2, 2.9e-2, 3e-2]
dimP = 4
dimQ = 4
str_ope = 'b2p_freqDep'

results = []

data = {
    'geometry1': geometry1,
    'geometry2': geometry2,
    'ope': str_ope,
    'lc': lc,
    'dimP': dimP,
    'dimQ': dimQ,
    'frequencies': freqvec.tolist(),
    'Z_center': None
}

for lc in tqdm(lc_list):
    mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

    dx, ds, _ = mesh_.integral_mesure()
    P, _ = mesh_.fonction_spaces()
    xref = mesh_.xref
    mesh = mesh_.mesh

    p, v = TrialFunction(P), TestFunction(P)

    f = inner(1, v) * ds(1)

    k = inner(grad(p), grad(v))
    m = inner(p, v)

    fx1 = Function(P)
    fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

    n   = FacetNormal(mesh)
    dp  = inner(grad(p), n) # dp/dn = grad(p) * n
    ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
    c_1 = inner(ddp, v)
    c_2 = inner(p, v)

    Pav1    = np.zeros(freqvec.size, dtype=np.complex128)
    Psol1 = Function(P)
    for ii in range(freqvec.size):
        freq = freqvec[ii]
        omega0 = 2*np.pi*freq
        k0 = omega0/c0
        d_0 = 2j*k0 + 4*fx1
        d_1 = 2*fx1**2 - k0**2 + 4j*k0*fx1
        coeff_1 = 1/d_0
        coeff_2 = d_1/d_0
        a = k * dx - k0**2 * m * dx + coeff_1*c_1*ds(3) + coeff_2*c_2*ds(3)

        problem = LinearProblem(a, f, petsc_options_prefix="abc_b2pFreqDep", petsc_options={"ksp_type": "gmres",
                                                                                            "pc_type": "lu",
                                                                                            "pc_factor_mat_solver_type": "mumps"})

        p_solution = problem.solve()
        Pav1[ii] = assemble_scalar(form(p_solution*ds(1)))

    surfarea = assemble_scalar(form(1*ds(1)))
    k_output = 2*np.pi*freqvec/c0
    Z_center = 1j*k_output* Pav1 / surfarea
    Z_ana = compute_analytical_radiation_factor(freqvec, radius)
    data['lc'] = lc
    data['Z_center'] = {
        'real': Z_center.real.tolist(),
        'imag': Z_center.imag.tolist()
    }
    filename = data['geometry1'] + "_" + data['geometry2'] + "_" + data['ope'] + "_" + str(data['lc']) + "_" + str(data['dimP']) + "_" + str(data['dimQ'])
    save_json(data, filename)
    rmse = np.sqrt(np.mean(np.abs(Z_center - Z_ana)**2))
    results.append(rmse)
#save_results(file_str, Z_center)



fig, ax = plt.subplots(figsize=(16,9))




ax.plot([1/lc for lc in lc_list], results, label='RMSE', marker='o')

ax.legend(loc='upper left')
ax.set_xlabel('1/lc')
ax.set_ylabel('RMSE')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True)
plt.show()



