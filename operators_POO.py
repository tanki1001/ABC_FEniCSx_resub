
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import scipy.linalg as la
from scipy import special
import json

import pyvista

from tqdm import tqdm
from time import time

from abc import ABC, abstractmethod

import gmsh
import dolfinx 
import basix
from dolfinx import plot, fem, default_scalar_type, cpp, geometry as dolf_geo
from basix.ufl import element
from dolfinx.io import gmshio
import dolfinx.mesh as msh
from mpi4py import MPI

from dolfinx.fem import Function, functionspace, assemble, form, petsc, Constant, assemble_scalar, Expression
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, div, inner, Measure, FacetNormal, CellNormal)
import ufl
import petsc4py
from petsc4py import PETSc
import slepc4py.SLEPc as SLEPc

rho0   = 1.21
c0     = 343.8
source = 1

class Mesh:
    

    def __init__(self, degP, degQ, side_box, radius, lc, geometry, model_name = "no_name"):
        '''
        Constructor of the class Mesh. A Mesh is created from a function implemented in the module geometries. This class is not perfect, it only implements geometries of this case.
        input : 
            side_box   = float : 
            radius     = float : 
            lc         = float :
            geometry   = fonction :
            model_name = str : 

        output : 
            Mesh

        '''
        self.degP = degP
        self.degQ = degQ
        
        mesh_info, submesh_info = geometry(side_box, radius, lc, model_name)

        self.mesh         = mesh_info[0]
        self.mesh_tags    = mesh_info[1]
        self.mesh_bc_tags = mesh_info[2]
        self.xref         = mesh_info[3]
        #self.ridge_tags   = mesh_info[4]

        self.submesh          = submesh_info[0]
        self.entity_maps_mesh = submesh_info[1]
        #self.edge_submesh     = edge_submesh
        self.lc = lc

    
    def integral_mesure(self):
        '''
        This function gives access to the integral operator over the mesh and submesh in Mesh instance
        input :

        output :
            dx  = Measure : integral over the whole domain
            ds  = Measure : integral over the tagged surfaces
            dx1 = Measure : integral over the whole subdomain

        '''
        mesh         = self.mesh
        mesh_tags    = self.mesh_tags
        mesh_bc_tags = self.mesh_bc_tags
    
        submesh = self.submesh

        dx  = Measure("dx", domain=mesh, subdomain_data=mesh_tags)
        ds  = Measure("ds", domain=mesh, subdomain_data=mesh_bc_tags)
        dx1 = Measure("dx", domain=submesh)
        
        
        return dx, ds, dx1

    def fonction_spaces(self, family = "Lagrange"):
        '''
        This function provide fonction spaces needed in the FEM. They are spaces where the test and trial functions will be declared.
        input : 
            family = str : family of the element

        output : 
            P = FunctionSpace : fonction space where the fonctions living in the acoutic domain will be declared
            Q = FonctionSpace : fonction space where the fonctions living in the subdomain will be declared
        '''
        degP    = self.degP
        degQ    = self.degQ 
        mesh    = self.mesh
        submesh = self.submesh
    
        P1 = element(family, mesh.basix_cell(), degP)
        P = functionspace(mesh, P1)

        Q1 = element(family, submesh.basix_cell(), degQ)
        Q = functionspace(submesh, Q1)
        
        return P, Q
    
class Simulation:

    def __init__(self, mesh, operator, loading):
        '''
        Constructor of the class Simulation. 
        input : 
            mesh     = Mesh
            operator = Operator
            loading  = Loading
        output :
            Simulation
        '''
        self.mesh     = mesh
        self.operator = operator
        self.loading  = loading

    def set_mesh(self, mesh):
        '''
        Setter to change the geometry on the one the same simulation will be run
        input : 
            mesh = Mesh : new Mesh obtained from a new geometry
        '''
        self.mesh = mesh

    def set_operator(self, ope):
        '''
        Setter to change the operator applied on the simulation
        input : 
            ope = Operator 
        '''
        self.operator = ope

    def set_loading(self, loading):
        '''
        Setter to change the loading applied on the simulation
        input : 
            loading = Loading 
        '''
        self.loading = loading
    
    # To edit
    def FOM(self, freq, frequency_sweep = True):
        """
        Function to select either a frequency sweep or a singular frequency computation
        input :
            freq = np.arange() or int : frequency interval or singular frequency
            frequency_sweep = bool : True if freq is a frequency interval, False if freq is a singular frequency
        """

        if frequency_sweep and not(isinstance(freq, int)):
            print('Frequency sweep')
            return self.freq_sweep_FOM_newVersion(freq)
        elif isinstance(freq, int) and not(frequency_sweep):
            print('Singular frequency')
            return self.singular_frequency_FOM(freq)
    

    def freq_sweep_FOM_newVersion(self, freqvec):
        '''
        Frequency sweep: solve [Z(k)] X = F(f) at each frequency.

        Returns:
            X_sol : list of Function(P) — pressure solution at each frequency
        '''
        
        ope    = self.operator
        list_D = ope.list_D
        mesh   = self.mesh
        loading = self.loading

        P, _   = mesh.fonction_spaces()
        offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

        F_static = loading.F_static
        F_static.assemble()
        freq_dep = loading.freq_dep_order > 0


        ksp = PETSc.KSP().create()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()

        X_sol = []
        for ii in tqdm(range(freqvec.size)):
            freq = freqvec[ii]
            k0 = 2 * np.pi * freq / c0

            Z = list_D[0]
            for i in range(1, len(list_D)):
                Z = Z + (1j * k0)**i * list_D[i]
            if ii == 0:
                print(f"Size of the global matrix: {Z.getSize()}")

            # Loading at this frequency
            if freq_dep:
                F_freq = loading.F_at_freq(freq)
                X = F_freq.copy()
            else:
                F_freq = F_static
                X = Z.createVecRight()


            ksp.setOperators(Z)
            #
            
            ksp.solve(F_freq, X)

            Psol = Function(P)
            Psol.x.array[:offset] = X.array_r[:offset]
            #print(f"Max pressure at f={freq} Hz: {np.max(np.abs(X.array_r[:offset]))}")
            X_sol.append(Psol)
            
            X.destroy()
            Z.destroy()
            if freq_dep:
                F_freq.destroy()

        return X_sol
    
    def singular_frequency_FOM(self, freq, scalar_sol=True):
        '''
        Solve [Z(k)] X = F(f) at a single frequency.

        Returns:
            Psol1, Qsol1 : Function(P), Function(Q) — if scalar_sol=True
            X             : PETSc.Vec — if scalar_sol=False
        '''
        ope    = self.operator
        list_D = ope.list_D
        mesh   = self.mesh
        loading = self.loading

        P, Q         = mesh.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

        ksp = PETSc.KSP().create()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        
        # Loading at this frequency
        if loading.freq_dep_order > 0:
            F = loading.F_at_freq(freq)
        else:
            F = loading.F_static
        F.assemble()

        k0 = 2 * np.pi * freq / c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j * k0)**i * list_D[i]
        print(f"Size of the global matrix: {Z.getSize()}")

        ksp.setOperators(Z)
        X = F.copy()
        ksp.solve(F, X)

        Psol1.x.array[:offset] = X.array_r[:offset]
        Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]

        ksp.destroy()
        Z.destroy()
        if loading.freq_dep_order > 0:
            F.destroy()

        harry_plotterv2([P, Q], [Psol1, Qsol1], ['p', 'q'], show_edges=False)

        if scalar_sol:
            X.destroy()
            return Psol1, Qsol1
        else:
            return X
        
    def wcawe_newVersion(self, N, freq, BSP=True):
        '''
        WCAWE model order reduction: builds an orthonormal projection basis Vn.

        Handles frequency-dependent loading via the first sum (F derivatives).
        For constant loading (Loading), the first sum is zero.
        For monopole loading (Loading_monopole), F^(1) = j·2π · F_static contributes.

        input :
            N    = int : nb of vectors in the projection basis
            freq = float : interpolation frequency (expansion point)
            BSP  = bool : apply Block-Structure-Preserving splitting

        output :
            Vn, CPUbuildingVn, CPUderivatives, CPUspliting_Vn
        '''
        ope    = self.operator
        list_D = ope.list_D
        degZ   = len(list_D)
        loading = self.loading

        F_static = loading.F_static
        F_static.assemble()
        freq_dep = loading.freq_dep_order > 0

        # Compute the jth derivative of Z
        list_Zj = []
        t1 = time()
        for j in range(degZ):
            Zj = None
            for i in range(j, degZ):
                coeff = (2j * np.pi / c0)**i * factorial(i) / factorial(i - j) * freq**(i - j)
                if Zj is None:
                    Zj = coeff * list_D[i]
                else:
                    Zj = Zj + coeff * list_D[i]
            list_Zj.append(Zj)
        t2 = time()
        CPUderivatives = t2 - t1
        print(f'Time to compute frequency derivatives: {CPUderivatives} secs')

        # F at expansion point
        if freq_dep:
            F_f0 = loading.F_at_freq(freq)
            F_coeffs = loading.F_deriv_coeffs(freq, loading.freq_dep_order)
        else:
            F_f0 = F_static

        # Solver (factorize Z^(0) once)
        ksp = PETSc.KSP().create()
        ksp.setOperators(list_Zj[0])
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        # Q matrix: norms (diagonal) and projections (upper triangular)
        Q_mat = PETSc.Mat().create()
        Q_mat.setSizes((N, N))
        Q_mat.setType("seqdense")
        Q_mat.setFromOptions()
        Q_mat.setUp()

        # First basis vector: v1 = Z^(0)^{-1} F(f0), normalized
        t1 = time()
        v1 = F_f0.copy()
        ksp.solve(F_f0, v1)

        norm_v1 = v1.norm()
        print(f'norm 1st vector : {norm_v1}')
        v1.normalize()
        Q_mat.setValue(0, 0, norm_v1)
        size_v1 = v1.getSize()

        # Basis matrix Vn (N_dofs x N)
        Vn = PETSc.Mat().create()
        Vn.setSizes((size_v1, N))
        Vn.setType("seqdense")
        Vn.setFromOptions()
        Vn.setUp()
        Vn.setValues(list(range(size_v1)), 0, v1, PETSc.InsertMode.INSERT_VALUES)

        if freq_dep:
            F_f0.destroy()

        for n in tqdm(range(2, N + 1)):
            # --- First sum: F derivative contributions ---
            rhs_first = list_Zj[0].createVecLeft()
            if freq_dep:
                for l in range(1, min(loading.freq_dep_order + 1, n)):
                    if F_coeffs[l] != 0:
                        # F^(l)(f0) = F_coeffs[l] * F_static
                        F_l = F_static.copy()
                        F_l.scale(F_coeffs[l])

                        # Z^(0)^{-1} * F^(l)
                        temp = list_Zj[0].createVecRight()
                        ksp.solve(F_l, temp)
                        F_l.destroy()

                        # Correction term P_Q_w for first sum (omega=1)
                        P_q_1 = P_Q_w(Q_mat, n, l, 1)
                        P_q_1_col = P_q_1.getColumnVector(n - l - 1)

                        # Multiply: temp is a single vector, scale by the correction scalar
                        # P_q_1_col is a vector of length (n-l), we need its last entry
                        # Actually we need V_{n-l} * P_q_1_col, same pattern as second sum
                        row_is = PETSc.IS().createStride(size_v1, first=0, step=1)
                        col_is = PETSc.IS().createStride(n - l, first=0, step=1)
                        # But wait — Z^(0)^{-1}*F^(l) is a single vector, not V*something
                        # The first sum formula is:
                        # sum_l Z0^{-1} * F^(l) * prod(omega_{n-j,n}) for j=0..l-1
                        # The correction prod(omega) is encoded in P_Q_w with omega=1
                        # P_q_1 is (n-l x n-l), P_q_1_col = P_q_1[:, n-l-1]
                        # The contribution is Z0^{-1}*F^(l) * (sum of P_q_1_col entries applied to basis)
                        # Actually, re-reading Eq. 13: the first sum doesn't involve V —
                        # it's Z0^{-1} * F^(l) * scalar_correction
                        # P_q_1_col has shape (n-l,), its last component is the scalar we need
                        p_q_1_vals = P_q_1_col.getArray()
                        # The correction is a scalar: last entry of P_q_1_col
                        scalar_corr = p_q_1_vals[-1] if len(p_q_1_vals) > 0 else 0.0
                        rhs_first.axpy(scalar_corr, temp)

                        row_is.destroy()
                        col_is.destroy()
                        temp.destroy()
                        P_q_1.destroy()
                        P_q_1_col.destroy()

            # --- Second sum: Z derivative contributions ---
            rhs_second = list_Zj[0].createVecLeft()
            for j in range(2, min(degZ, n)):
                P_q_2        = P_Q_w(Q_mat, n, j, 2)
                P_q_2_values = P_q_2.getColumnVector(n - j - 1)

                row_is = PETSc.IS().createStride(size_v1, first=0, step=1)
                col_is = PETSc.IS().createStride(n - j, first=0, step=1)

                Vn_i       = Vn.createSubMatrix(row_is, col_is)
                Vn_i       = list_Zj[j].matMult(Vn_i)
                Vn_i_P_q_2 = Vn_i.createVecLeft()
                Vn_i.mult(P_q_2_values, Vn_i_P_q_2)

                rhs_second.axpy(1.0, Vn_i_P_q_2)

                row_is.destroy()
                col_is.destroy()
                P_q_2.destroy()
                P_q_2_values.destroy()
                Vn_i.destroy()
                Vn_i_P_q_2.destroy()

            # Z^(1) * v_{n-1}
            rhs_Z1 = list_Zj[0].createVecLeft()
            vn_1 = Vn.getColumnVector(n - 2)
            list_Zj[1].mult(vn_1, rhs_Z1)

            # RHS = first_sum - Z^(1)*v_{n-1} - second_sum
            rhs = rhs_first.copy()
            rhs.axpy(-1.0, rhs_Z1)
            rhs.axpy(-1.0, rhs_second)

            vn = Vn.createVecLeft()
            ksp.solve(rhs, vn)
            rhs.destroy()
            rhs_first.destroy()
            rhs_Z1.destroy()
            rhs_second.destroy()

            # Modified Gram-Schmidt: store coefficients in Q and orthogonalize in one pass
            for i in range(n - 1):
                v_i = Vn.getColumnVector(i)
                coeff = vn.dot(v_i)
                Q_mat.setValue(i, n - 1, coeff)
                vn = vn - coeff * v_i
                v_i.destroy()

            norm_vn = vn.norm()
            Q_mat.setValue(n - 1, n - 1, norm_vn)
            Q_mat.assemble()

            vn.normalize()
            Vn.setValues(list(range(size_v1)), n - 1, vn, PETSc.InsertMode.INSERT_VALUES)

        t2 = time()
        CPUbuildingVn = t2 - t1
        print("Vn basis has been built")
        print(f"CPU_time_vec : \n{CPUbuildingVn}")
        ksp.destroy()
        Vn.assemble()
        if BSP:
            Vn_tilde, CPUspliting_Vn = self.split_Vn(Vn)
            return Vn_tilde, CPUbuildingVn, CPUderivatives, CPUspliting_Vn
        else:
            return Vn, CPUbuildingVn, CPUderivatives, 0
    
    def split_Vn(self, Vn):

        P, Q        = self.mesh.fonction_spaces()
        offset      = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far

        (len_rows, len_col) = Vn.getSize()

        ### Create the empty basis
        Vn_tilde = PETSc.Mat().create()
        Vn_tilde.setSizes((len_rows, 2*len_col))  
        Vn_tilde.setType("seqdense")  
        Vn_tilde.setFromOptions()
        Vn_tilde.setUp()
        t1 = time()
        for i in range(len_col):
            vi = Vn.getColumnVector(i)
            vp = vi[:offset]
            vq = vi[offset:]
            
            gram_shmidt_wo0 = True
            if i > 0 and gram_shmidt_wo0 :
                #print("gram_shmidt - Orthonormalization of the subbasis on them self")
                for ii in range(i):
                    vi_1 = Vn.getColumnVector(ii)
                    vp_1 = vi_1[:offset]
                    vq_1 = vi_1[offset:]
                    vp = vp - np.vdot(vp_1, vp)*vp_1
                    vq = vq - np.vdot(vq_1, vq)*vq_1
            #print(f'After GS norm vp{i} : {np.linalg.norm(vp)}')
            #print(f'After GS norm vq{i} : {np.linalg.norm(vq)}')

            vp = vp / np.linalg.norm(vp)
            vq = vq / np.linalg.norm(vq)
            Vn_tilde.setValues([i for i in range(offset)], i, vp, PETSc.InsertMode.INSERT_VALUES)
            Vn_tilde.setValues([i for i in range(offset, len_rows)], len_col + i, vq, PETSc.InsertMode.INSERT_VALUES)
        
        Vn_tilde.assemble()
        t2 = time()
        CPUspliting_Vn = t2-t1
        print(f"Time to split the basis : {CPUspliting_Vn} secs")
        
        return Vn_tilde, CPUspliting_Vn
    
    def merged_WCAWE(self, list_N, list_freq, BSP = True):
        if len(list_N) != len(list_freq):
            print(f"WARNING : The list of nb vector values and the list of interpolated frequencies does not match: {len(list_N)} - {len(list_freq)}")

        size_Vn = sum(list_N)
        V1, CPUbuildingVn, CPUderivatives, CPUspliting_Vn = self.wcawe_newVersion(list_N[0], list_freq[0], BSP)
        if False:
            list_CPU_time_vec.append(CPU_time_vec)
            size_V1 = V1.getSize()[0]
            Vn = PETSc.Mat().create()
            Vn.setSizes((size_V1, size_Vn))
            Vn.setType("seqdense")  
            Vn.setFromOptions()
            Vn.setUp()    
            for i in range(V1.getSize()[0]):
                for j in range(V1.getSize()[1]):
                    Vn[i, j] = V1[i, j]
            count = list_N[0]
            for i in range (1,len(list_freq)):
                Vi, CPU_time_vec = self.wcawe_newVersion(list_N[i], list_freq[i])
                list_CPU_time_vec.append(CPU_time_vec)
                for ii in range(Vi.getSize()[0]):
                    for jj in range(Vi.getSize()[1]):
                        Vn[ii, count + jj] = Vi[ii, jj]
                count += list_N[i]
            Vn.assemble()
            print_CPU_vec = False
            if print_CPU_vec:
                fig, axs = plt.subplots(nrows=1, ncols=len(list_N), figsize = (16, 9))
                for i in range(len(list_N)):
                    if len(list_N) == 1:
                        axs = [axs]
                    axs[i].plot(range(2, list_N[i]+1), list_CPU_time_vec[i])
                plt.grid()
                plt.xlabel("Vector")
                plt.ylabel("CPU time (s)")
                plt.title("CPU time per vector")
                plt.show()
            
        return V1, CPUbuildingVn, CPUderivatives, CPUspliting_Vn

    def moment_matching_MOR(self, Vn, freqvec):
        '''
        Frequency sweep of the reduced model: Zr(k) * alpha = Fr(f), then X = Vn * alpha.

        Handles frequency-dependent loading by scaling Fr at each frequency.

        Returns:
            X_sol      : list of Function(P) — pressure solution at each frequency
            solvingMOR : float — CPU time for the ROM sweep
        '''
        ope    = self.operator
        list_D = ope.list_D
        mesh_  = self.mesh
        loading = self.loading

        P, _   = mesh_.fonction_spaces()
        offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

        F_static = loading.F_static
        F_static.assemble()
        freq_dep = loading.freq_dep_order > 0

        # Project matrices and F_static
        list_Dr, Fr_static = listDr_matrices(list_D, F_static, Vn)

        X_sol = []
        t1 = time()
        for ii in tqdm(range(freqvec.size)):
            freq = freqvec[ii]
            k0 = 2 * np.pi * freq / c0

            Zr = list_Dr[0]
            for i in range(1, len(list_Dr)):
                Zr = Zr + (1j * k0)**i * list_Dr[i]
            if ii == 0:
                print(f"Size of the global reduced matrix: {Zr.getSize()}")

            Zr.convert("seqaij")

            # Fr at this frequency
            if freq_dep:
                Fr = Fr_static.copy()
                Fr.scale(loading.F_deriv_coeffs(freq, 0)[0])
            else:
                Fr = Fr_static

            ksp = PETSc.KSP().create()
            ksp.setOperators(Zr)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")

            alpha = Fr.copy()
            ksp.solve(Fr, alpha)

            X = F_static.copy()
            Vn.mult(alpha, X)

            Psol = Function(P)
            Psol.x.array[:offset] = X.array_r[:offset]
            X_sol.append(Psol)

            ksp.destroy()
            alpha.destroy()
            X.destroy()
            Zr.destroy()
            if freq_dep:
                Fr.destroy()

        t2 = time()
        solvingMOR = t2 - t1
        print(f'Time to compute Zralpha = Fr : {solvingMOR}')
        return X_sol, solvingMOR

    def compute_radiation_factor(self, freqvec, X_sol):
        """Compute the radiation factor from a list of solution Functions.

        Args:
            freqvec : array of frequencies
            X_sol   : list of Function(P), one per frequency

        Returns:
            Z_center : complex array — radiation factor at each frequency
        """
        _, ds, _ = self.mesh.integral_mesure()
        surfarea = assemble_scalar(form(1*ds(1)))
        k_output = 2*np.pi*freqvec/c0

        Pav = np.zeros(freqvec.size, dtype=np.complex128)
        for ii, Xf in enumerate(X_sol):
            Pav[ii] = assemble_scalar(form(Xf * ds(1)))

        Z_center = 1j*k_output * Pav / surfarea
        return Z_center

    def plot_radiation_factor(self, ax, freqvec, X_sol, s = '', dashed = False, compute = True):
        """Plot the radiation factor.

        Args:
            X_sol   : list of Function(P) if compute=True, or pre-computed Z_center array if compute=False
        """
        if compute :
            Z_center = self.compute_radiation_factor(freqvec, X_sol)
        else:
            Z_center = X_sol
        if dashed:
            ax.plot(freqvec, Z_center.real, label = s, linestyle = "--")
        else :
            ax.plot(freqvec, Z_center.real, label = s)
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'$\sigma$')
        ax.set_title(f'Order : {self.operator.orderOperator}')

    def evaluate_at_point(self, X_sol, freqvec, x0):
        """Evaluate pressure at point x0 for each solution in X_sol.

        Args:
            X_sol   : list of Function(P), one per frequency
            freqvec : array of frequencies (used only for NaN warning messages)
            x0      : array-like of shape (3,) — evaluation point coordinates

        Returns:
            u_values : complex array of shape (len(X_sol),)
        """
        mesh = self.mesh.mesh
        bb_tree = dolf_geo.bb_tree(mesh, mesh.topology.dim)
        points = np.array([x0]).T
        cell_candidates = dolf_geo.compute_collisions_points(bb_tree, points.T)
        colliding_cells = dolf_geo.compute_colliding_cells(mesh, cell_candidates, points.T)

        points_on_proc = []
        cells = []
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)

        u_values = []
        for ii, Xf in enumerate(X_sol):
            val = Xf.eval(points_on_proc, cells)
            if np.isnan(val[0]):
                print(f"NaN value for {freqvec[ii]}Hz")
            u_values.append(val[0])
        return np.array(u_values)

class Operator(ABC):
    '''
    This method aims at being an abstract one. One will definied an implemented operator, and never use the followong constructor.
    '''
    def __init__(self, mesh):
        '''
        Constructor of the class Operator. The objective is to overwrite this constructor in order for the user to only use designed Operator.
        input : 
            mesh = Mesh : instance of the class Mesh, where the Operator is applied on

        output :
            Operator
        '''

        self.mesh        = mesh
        #self.list_Z      = None
        #self.list_coeffZ = None
        self.list_D      = None

    @abstractmethod
    def import_matrix(self, freq):
        pass

    # This method, using an old way to assemble matrices, serves the method Simulation.plot_sv_listZ
    @abstractmethod
    def get_listZ(self):
        pass

class B1p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b1p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b1p()
        self.list_D = self.b1p_newVersion()

  
    # To edit
    def b1p_newVersion(self):
        '''
        Create all the constant Form of the b1p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        submesh      = self.mesh.submesh
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
    
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(P)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        g1 = inner(fx1*p, u)*ds(3)
        g2 = inner(p, u)*ds(3)
        e  = inner(q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        G1 = form(g1, entity_maps=entity_maps_mesh)
        G2 = form(g2, entity_maps=entity_maps_mesh)
        
        E = form(e)

        D1 = [[K,  C],
              [G1, E]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1, entity_maps=entity_maps_mesh)
        D2 = [[D2_00, D2_01],
              [G2,    D2_11]]
        D2 = petsc.assemble_matrix_block(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3_10 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, u) * ds(3), entity_maps=entity_maps_mesh)
        D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D3 = [[M,     D3_01],
              [D3_10, D3_11]]
        D3 = petsc.assemble_matrix_block(D3)
        D3.assemble()
    
        list_D       = np.array([D1, D2, D3])
    
        return list_D
    
    def get_listZ(self):
        pass

    def import_matrix(self, freq):
        pass

class B2p(Operator):

    def __init__(self, mesh):
        '''

        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()
        self.orderOperator = 1
        

  
    # To edit
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(-q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(4*fx1_p*p, u)*ds(3)
        g4  = inner(2*fx1_p**2*p, u)*ds(3)

        e1  = inner(4*fx1_q*q, u)*dx1
        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K,     C],
              [G1_4, E1]]
        D1 = petsc.assemble_matrix(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2 = [[D2_00,   D2_01],
              [G3,         E2]]
        D2 = petsc.assemble_matrix(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D3 = [[M,  D3_01],
              [G2, D3_11]]
        D3 = petsc.assemble_matrix(D3)
        D3.assemble()
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D


    def get_listZ(self):
        pass
    
    
    def import_matrix(self, freq):
        pass

class B2p_tang(Operator):

    def __init__(self, mesh):
        '''

        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()
        self.orderOperator = 1
        

  
    # To edit
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh) # Normal to the boundaries
        #curv_q = ufl.div(ns)
        #curv_q = 2/0.11
        #print(type(curv_q))

        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 


        kappa = ufl.div(n)

        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        exterior_boundaries = dolfinx.mesh.exterior_facet_indices(mesh.topology)

        submesh_info = [submesh, entity_maps_mesh]
        curv_q = move_to_facet_quadrature(kappa, mesh, submesh_info)
        #curv_q = Function(Q)
        #curv_q.interpolate(curv_q1)

        #curv_q = Constant(submesh, PETSc.ScalarType(20))
        #deg        = self.deg
        
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        #fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0])**2 + (x[1])**2 + (x[2])**2))

        fx1_q = Function(Q)
        #fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0])**2 + (x[1])**2 + (x[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(-q, v)*ds(3)
        
        gradt_p    = tangential_proj(grad(p), n) 
        gradt_u    = tangential_proj(grad(u), n)         
        surf_lap_p = ufl.div(gradt_p)

        g1  = inner(-surf_lap_p, u)*ds(3)
        g2  = inner(2*p, u)*ds(3)
        g3  = inner(4*fx1_p*p, u)*ds(3)
        g4  = inner(2*fx1_p**2*p, u)*ds(3)

        e1  = inner((4*fx1_q-curv_q)*q, u)*dx1
        
        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1, entity_maps=entity_maps_mesh)
        E2 = form(e2)


        D1 = [[K,     C],
            [G1_4, E1]]
        D1 = petsc.assemble_matrix(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2 = [[D2_00,   D2_01],
            [G3,         E2]]
        D2 = petsc.assemble_matrix(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D3 = [[M,  D3_01],
            [G2, D3_11]]
        D3 = petsc.assemble_matrix(D3)
        D3.assemble()
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D
    
    def import_matrix(self, freq):
        pass
    def get_listZ(self):
        pass

class B2p_tang_ipp(Operator):

    def __init__(self, mesh):
        '''

        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()
        self.orderOperator = 1



    # To edit
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff

        input :
            mesh_info    = List[]
            submesh_info = List[]

        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol

        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh) # Normal to the boundaries
        #curv_q = ufl.div(ns)
        #curv_q = 2/0.11
        #print(type(curv_q))

        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure()
        #dl = Measure("dr", domain=mesh, subdomain_data=self.mesh.ridge_tags)

        kappa = ufl.div(n)

        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        exterior_boundaries = dolfinx.mesh.exterior_facet_indices(mesh.topology)

        submesh_info = [submesh, entity_maps_mesh]
        curv_q = move_to_facet_quadrature(kappa, mesh, submesh_info)

        #curv_q = Function(Q)
        #curv_q.interpolate(curv_q1)

        #curv_q = Constant(submesh, PETSc.ScalarType(20))
        #deg        = self.deg
        
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        #fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0])**2 + (x[1])**2 + (x[2])**2))

        fx1_q = Function(Q)
        #fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0])**2 + (x[1])**2 + (x[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(-q, v)*ds(3)
        
        # Integration by parts of -Δ_s p:
        # ∫_Γ (-Δ_s p) u dΓ = ∫_Γ ∇_s p · ∇_s u dΓ  (boundary term vanishes for closed Γ)
        #
        # FFCx cannot tabulate 3D basis functions on 2D submesh quadrature points,
        # so we assemble K_surf (Q×Q surface stiffness) on the submesh and compose
        # with an interpolation matrix I_PQ that maps P boundary DOFs to Q DOFs:
        #   G1_contribution = K_surf @ I_PQ
        g1_surf = inner(grad(q), grad(u)) * dx1  # Q×Q surface stiffness
        g2  = inner(2*p, u)*ds(3)
        g3  = inner(4*fx1_p*p, u)*ds(3)
        g4  = inner(2*fx1_p**2*p, u)*ds(3)

        e1  = inner((4*fx1_q-curv_q)*q, u)*dx1

        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(c, entity_maps=entity_maps_mesh)

        G4 = form(g4, entity_maps=entity_maps_mesh)
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)

        E1 = form(e1, entity_maps=entity_maps_mesh)
        E2 = form(e2)

        # --- Assemble D1 with G4 in block (1,0); G1 will be added manually ---
        D1 = [[K,   C],
              [G4, E1]]
        D1 = petsc.assemble_matrix(D1)
        D1.assemble()

        # --- Build G1 contribution: K_surf @ I_PQ ---
        # Step 1: Surface stiffness matrix K_surf (nQ × nQ) on submesh
        K_surf = petsc.assemble_matrix(form(g1_surf))
        K_surf.assemble()

        # Step 2: Interpolation matrix I_PQ (nQ × nP)
        # Each Q DOF maps to the P DOF at the same geometric location
        from scipy.spatial import cKDTree
        Q_coords = Q.tabulate_dof_coordinates()
        P_coords = P.tabulate_dof_coordinates()
        nQ_local = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
        nP_local = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

        tree = cKDTree(P_coords[:nP_local])
        dists, p_idx = tree.query(Q_coords[:nQ_local])

        I_PQ = PETSc.Mat().create()
        I_PQ.setSizes((nQ_local, nP_local))
        I_PQ.setType("aij")
        I_PQ.setPreallocationNNZ(1)
        I_PQ.setUp()
        for i in range(nQ_local):
            I_PQ.setValue(i, int(p_idx[i]), 1.0)
        I_PQ.assemble()

        # Step 3: G1_mat = K_surf @ I_PQ (nQ × nP)
        G1_mat = K_surf.matMult(I_PQ)

        # Step 4: Add G1_mat to D1 at block (1,0) position (rows offset by nP)
        offset = nP_local
        D1.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        for i in range(nQ_local):
            cols, vals = G1_mat.getRow(i)
            if len(cols) > 0:
                D1.setValues([i + offset], cols.tolist(), vals.tolist(),
                             PETSc.InsertMode.ADD_VALUES)
        D1.assemble()

        K_surf.destroy()
        I_PQ.destroy()
        G1_mat.destroy()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2 = [[D2_00,   D2_01],
            [G3,         E2]]
        D2 = petsc.assemble_matrix(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D3 = [[M,  D3_01],
            [G2, D3_11]]
        D3 = petsc.assemble_matrix(D3)
        D3.assemble()
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D
    
    def import_matrix(self, freq):
        pass
    def get_listZ(self):
        pass

class Loading_abstract(ABC):
    '''
    Abstract base class for loadings. Subclasses must implement assemble_F().

    The loading can be frequency-dependent: F(f) = scale(f) * F_static.
    For a constant loading, scale(f) = 1 (default).
    For a monopole, scale(f) = j*2*pi*f.
    '''

    def __init__(self, mesh):
        self.mesh = mesh

    @abstractmethod
    def assemble_F(self):
        '''Assemble the frequency-independent part of the loading. Returns PETSc.Vec.'''
        pass

    @property
    def freq_dep_order(self):
        '''Max derivative order of F w.r.t. frequency. 0 = constant, 1 = linear in f.'''
        return 0

    def F_at_freq(self, freq):
        '''Return F(f) as a PETSc.Vec. Default: F_static (constant loading).'''
        return self.F_static

    def F_deriv_coeffs(self, f0, max_order):
        '''Return list of scalar coefficients [c0, c1, ..., c_max_order] such that
        F^(l)(f0) = c_l * F_static. Default: [1, 0, 0, ...] (constant loading).'''
        coeffs = [0.0] * (max_order + 1)
        coeffs[0] = 1.0
        return coeffs

class Loading(Loading_abstract):
    '''
    Vibrating plate loading: F = ∫ 1·v ds(1), constant in frequency.
    '''

    def __init__(self, mesh):
        super().__init__(mesh)
        self.F_static = self.assemble_F()
        self.F = self.F_static  # backward compat with standalone soar()/wcawe()

    def assemble_F(self):
        submesh = self.mesh.submesh
        P, Q    = self.mesh.fonction_spaces()
        _, ds, dx1 = self.mesh.integral_mesure()
        v, u = TestFunction(P), TestFunction(Q)

        f    = form(inner(1, v) * ds(1))
        zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

        F = petsc.assemble_vector([f, zero])
        return F

class Loading_monopole(Loading_abstract):
    '''
    Monopole point-source loading: F(f) = j·2πf · ρ₀·Q₀·δ(x-xs).

    The frequency-independent part F_static = ρ₀·Q₀·δ(x-xs) is assembled once.
    At each frequency: F(f) = j·2πf · F_static.
    '''

    def __init__(self, mesh, Q):
        super().__init__(mesh)
        self.Q_amplitude = Q
        self.F_static = self.assemble_F()
        self.F = self.F_static  # backward compat

    def assemble_F(self):
        '''Assemble F_static = ρ₀·Q₀·δ(x-xs) using point evaluation of basis functions.

        Precomputes cell contributions (scells, basis_values) and stores them
        for reuse in F_at_freq(). The returned F_static is a monolithic PETSc
        vector (compatible with MOR projection in listDr_matrices).
        '''
        P, Q = self.mesh.fonction_spaces()
        submesh = self.mesh.submesh
        dx, _, dx1 = self.mesh.integral_mesure()
        v, u = TestFunction(P), TestFunction(Q)

        # Precompute cell contributions for point source evaluation
        self.scells, self.basis_values = self.compute_cell_contributions(P)

        # Zero Q-block function (reused across F_at_freq calls)
        self.zero_Q = fem.Function(Q)
        self.zero_Q.x.array[:] = 0

        # Create monolithic F_static with correct block layout
        f_zero    = form(inner(Constant(self.mesh.mesh, PETSc.ScalarType(0)), v) * dx)
        zero_form = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)
        F = petsc.assemble_vector([f_zero, zero_form])

        # Fill P-block with rho0 * Q * phi_i(x_s) via point evaluation
        for cell, bv in zip(self.scells, self.basis_values):
            dofs = P.dofmap.cell_dofs(cell)
            bv = np.squeeze(bv)
            F.array[dofs] += bv * rho0 * self.Q_amplitude
        F.assemble()
        return F

    @property
    def freq_dep_order(self):
        return 1  # F(f) = j*2πf * F_static

    def F_at_freq(self, freq):
        '''Return F(f) = j·ω·ρ₀·Q·δ(x-xs) as a nested PETSc vector.

        Builds b1 = b_static * (j*k0*c0) where b_static contains ρ₀·Q·φ_i(x_s),
        matching the inline pattern used in freq_sweep_FOM_newVersion.
        '''
        P, _ = self.mesh.fonction_spaces()
        k0 = 2 * np.pi * freq / c0

        b1 = fem.Function(P)
        b1.x.array[:] = 0
        for cell, bv in zip(self.scells, self.basis_values):
            dofs = P.dofmap.cell_dofs(cell)
            bv = np.squeeze(bv)
            b1.x.array[dofs] += bv * (1j * rho0 * k0 * c0 * self.Q_amplitude)

        F_freq = PETSc.Vec().createNest([b1.x.petsc_vec, self.zero_Q.x.petsc_vec])
        return F_freq

    def F_deriv_coeffs(self, f0, max_order):
        '''F(f) = j·2πf · F_static → F^(0)(f0) = j·2πf0, F^(1) = j·2π, F^(l≥2) = 0.'''
        coeffs = [0.0] * (max_order + 1)
        coeffs[0] = 1j * 2 * np.pi * f0
        if max_order >= 1:
            coeffs[1] = 1j * 2 * np.pi
        return coeffs
    
    def compute_cell_contributions(self, V):
        
        x_ref = self.mesh.xref

        
        # Determine what process owns a point and what cells it lies within
        mesh = V.mesh
        points = np.array([x_ref], dtype=mesh.geometry.x.dtype)
        point_ownership_data = cpp.geometry.determine_point_ownership(
                mesh._cpp_object, points, 1e-6)
        owning_points = np.asarray(point_ownership_data.dest_points).reshape(-1, 3)
        cells = point_ownership_data.dest_cells

        # Pull owning points back to reference cell
        mesh_nodes = mesh.geometry.x
        cmap = mesh.geometry.cmap
        ref_x = np.zeros((len(cells), mesh.geometry.dim),
                        dtype=mesh.geometry.x.dtype)
        for i, (point, cell) in enumerate(zip(owning_points, cells)):
                geom_dofs = mesh.geometry.dofmap[cell]
                ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

        # Create expression evaluating a trial function (i.e. just the basis function)
        u = ufl.TrialFunction(V)
        num_dofs = V.dofmap.dof_layout.num_dofs * V.dofmap.bs
        if len(cells) > 0:
                # NOTE: Expression lives on only this communicator rank
                expr = Expression(u, ref_x, comm=MPI.COMM_SELF)
                values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))

                # Strip out basis function values per cell
                basis_values = values[:num_dofs:num_dofs*len(cells)]
        else:
                basis_values = np.zeros(
                (0, num_dofs), dtype=default_scalar_type)
        return cells, basis_values




###################################################
### Auxiliary functions ###
###################################################

### For operators ###
def tangential_proj(u, n):
    return (ufl.Identity(n.ufl_shape[0]) - ufl.outer(n, n)) * u


### For WCAWE ###

def sub_matrix(Q, start, end):
    '''
    This function is to obtain the sub matrix need for the correction term (P_q_w)
    intput :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        start = int : start index, reality index
        end   = int : end index, reality index

    output : 
        submatrix = np.array() : sub matrix, as a numpy matrix, because the size will remain low
    '''
     
    row_is    = PETSc.IS().createStride(end  - start + 1, first=start - 1, step=1)
    col_is    = PETSc.IS().createStride(end - start + 1, first=start - 1, step=1)
    submatrix = Q.createSubMatrix(row_is, col_is)

    row_is.destroy()
    col_is.destroy()

    submatrix = submatrix.getValues([i for i in range(end - start+1)], [i for i in range(end - start+1)])
    return submatrix

def P_Q_w(Q, alpha, beta, omega):
    '''
    Correction term function.
    input :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        alpha = int : reality value
        beta  = int : reality value
        omega = int : starting point of the product

    output :
        P_q_w = PETScMatType : correction term
    '''
    
    P_q = np.identity(alpha - beta) #create the identity matrix M*M with M = alpha - beta

    for t in range(omega, beta+1):
        sub_Q = sub_matrix(Q, t, alpha - beta + t - 1)
        sub_Q = np.linalg.inv(sub_Q)
        P_q   = np.dot(P_q, sub_Q)

    # The following lignes convert the result to a PETSc type
    P_q_w = PETSc.Mat().create()
    P_q_w.setSizes(P_q.shape, P_q.shape)
    P_q_w.setType("seqdense")  
    P_q_w.setFromOptions()
    P_q_w.setUp()

    for i in range(P_q.shape[0]):
        P_q_w.setValues(i, [j for j in range(P_q.shape[1])], P_q[i], PETSc.InsertMode.INSERT_VALUES)   
    P_q_w.assemble()
    return P_q_w

def listDr_matrices(list_D, F, Vn) :
    t1 = time()
    list_Dr = []
    # Compute the reduced matrices
    Vn_T = Vn.duplicate()
    Vn.copy(Vn_T)
    Vn_T.hermitianTranspose()
    Vn_T.assemble()
    print(f'Vn size : {Vn.getSize()}')
    print(f'Vn_T size : {Vn_T.getSize()}')

    for Di in list_D:
        Dir_m = Vn_T.matMult(Di) 
        Dir = Dir_m.matMult(Vn)
        Dir.assemble()
        list_Dr.append(Dir)
        Dir_m.destroy()

    Fr = list_Dr[0].createVecLeft()
    Vn_T.mult(F, Fr) # Fn = Vn_T * F
    #plot_heat_map(list_Dr[0])
    #plot_heat_map(list_Dr[1])
    #plot_heat_map(list_Dr[2])
    t2 = time()
    print(f'Time to reduce by projection the matrices :{t2-t1}')
    Vn_T.destroy()
    return list_Dr, Fr

### For SOAR ###

def soar(simu : Simulation, f0 : float, n : int):
    """
    SOAR procedure (Algorithm 4 from Bai & Su, SIAM J. Matrix Anal. Appl., 2005)
    adapted for model order reduction of the parametric system:

        [D1 + ik*D2 - k^2*D3] u(k) = F

    Expanding around k0 = 2*pi*f0/c0, the system becomes:
        Z(k) = A0 + dk*A1 + dk^2*A2    (dk = k - k0)
    where:
        A0 = D1 + ik0*D2 - k0^2*D3      (system matrix at expansion point)
        A1 = i*D2 - 2*k0*D3             (dZ/dk evaluated at k0)
        A2 = -D3                          (0.5 * d^2Z/dk^2)

    The SOAR algorithm builds an orthonormal basis Q of the second-order
    Krylov subspace G_n(A, B; u) where:
        A = -A0^{-1} * A1
        B = -A0^{-1} * A2 = A0^{-1} * D3
    without explicitly forming A or B. Each SOAR step requires only one
    linear solve with A0 (factorized once via MUMPS).

    The starting vector is u = A0^{-1} * F (solution at the expansion point),
    which ensures the reduced model matches the full-order model at k0.

    Args:
        simu : Simulation object (provides operator.list_D and loading.F)
        f0   : Interpolation frequency in Hz
        n    : Number of basis vectors to generate

    Returns:
        Vn            : PETSc.Mat (N_dofs x n, seqdense) orthonormal basis matrix
        CPU_basis     : float — time to build the basis (seconds)
        CPU_derivs    : float — time to compute shifted matrices (seconds)
        CPU_split     : float — always 0 (no BSP in SOAR), for interface consistency
    """
    print("start of SOAR procedure...")

    # --- Extract system components ---
    k0 = 2 * np.pi * f0 / c0
    [D1, D2, D3] = simu.operator.list_D
    loading = simu.loading
    if loading.freq_dep_order > 0:
        F = loading.F_at_freq(f0)
    else:
        F = loading.F_static
    F.assemble()

    # --- Build shifted system matrices ---
    t_deriv_start = time()
    # A0 = D1 + (ik0)*D2 + (ik0)^2*D3 = D1 + ik0*D2 - k0^2*D3
    A0 = D1 + (1j * k0) * D2 + ((1j * k0) ** 2) * D3
    # A1 = dZ/dk|_{k0} = i*D2 - 2*k0*D3
    A1 = 1j * D2 + (-2 * k0) * D3
    t_deriv_end = time()
    CPU_derivs = t_deriv_end - t_deriv_start
    print(f"SOAR: shifted matrices computed in {CPU_derivs:.4f} s")

    # --- Setup KSP solver (factorize A0 once, reuse for all solves) ---
    ksp = PETSc.KSP().create()
    ksp.setOperators(A0)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    # --- Alg.4, Step 1: q1 = u / ||u||  with u = A0^{-1} * F ---
    t_basis_start = time()
    #u = F.copy() #-> zsh: segmentation fault | The copy method gives a segmentation error
    u = A0.createVecRight()

    ksp.solve(F, u)
    u_norm = u.norm(PETSc.NormType.NORM_2)
    u.scale(1.0 / u_norm)

    N_dofs = u.getSize()
    row_indices = [i for i in range(N_dofs)]

    # --- Initialize basis matrix Vn (dense, N_dofs x n) ---
    Vn = PETSc.Mat().create()
    Vn.setSizes((N_dofs, n))
    Vn.setType("seqdense")
    Vn.setFromOptions()
    Vn.setUp()
    Vn.setValues(row_indices, 0, u.getArray(), PETSc.InsertMode.INSERT_VALUES)

    # --- Alg.4, Step 2: f = 0 ---
    f_vec = u.duplicate()
    f_vec.set(0.0)

    # Hessenberg matrix T storage
    # T_cols[j] stores column (j+1) of T-hat: [t_{1,j+1}, t_{2,j+1}, ..., t_{j+2,j+1}]
    T_cols = []

    # --- Alg.4, Step 3: Main SOAR loop ---
    for j in tqdm(range(n - 1), desc="SOAR basis construction"):
        qj = Vn.getColumnVector(j)

        # Step 4: r = A*q_j + B*f
        # Since A = -A0^{-1}*A1 and B = A0^{-1}*D3, this is equivalent to:
        #   A0 * r = -A1*q_j + D3*f
        rhs = qj.duplicate()
        temp = f_vec.duplicate()

        A1.mult(qj, rhs)            # rhs = A1 * q_j
        D3.mult(f_vec, temp)         # temp = D3 * f
        rhs.scale(-1.0)              # rhs = -A1 * q_j
        rhs.axpy(1.0, temp)          # rhs = -A1*q_j + D3*f

        r = qj.duplicate()
        ksp.solve(rhs, r)            # A0 * r = rhs  =>  r = A0^{-1} * rhs

        rhs.destroy()
        temp.destroy()
        qj.destroy()

        # Steps 5-8: Gram-Schmidt orthogonalization
        t_col = []
        for i in range(j + 1):
            qi = Vn.getColumnVector(i)
            tij = r.dot(qi)          # tij = qi^H * r  (Hermitian inner product)
            t_col.append(tij)
            r.axpy(-tij, qi)         # r = r - tij * qi
            qi.destroy()

        # Step 9: Compute norm
        t_j1_j = r.norm(PETSc.NormType.NORM_2)
        t_col.append(t_j1_j)
        T_cols.append(t_col)

        # Steps 10-18: Normal case / deflation / breakdown
        if t_j1_j > 1e-10:
            # Normal case (steps 11-12)
            r.scale(1.0 / t_j1_j)
            Vn.setValues(row_indices, j + 1, r.getArray(),
                         PETSc.InsertMode.INSERT_VALUES)
        else:
            # Deflation (steps 14-17): reset t_{j+1,j} = 1, q_{j+1} = 0
            T_cols[-1][-1] = 1.0
            print(f"  SOAR: deflation at step {j + 1}")
            # Column j+1 of Vn stays zero (initialized to zero by seqdense)

        r.destroy()

        # Step 12/16: Compute f = Q_j * T_hat(2:j+2, 1:j+1)^{-1} * e_{j+1}
        # T_hat(2:j+2, 1:j+1) is a (j+1) x (j+1) upper triangular matrix
        j_size = j + 1
        T_hat_sub = np.zeros((j_size, j_size), dtype=np.complex128)
        for col_idx in range(j_size):
            t_c = T_cols[col_idx]
            for row_idx in range(col_idx + 1):  # upper triangular: row <= col
                T_hat_sub[row_idx, col_idx] = t_c[row_idx + 1]

        e_vec = np.zeros(j_size, dtype=np.complex128)
        e_vec[-1] = 1.0
        x_coeff = np.linalg.solve(T_hat_sub, e_vec)

        # f = sum_i x_coeff[i] * q_{i+1}
        f_vec.set(0.0)
        for i in range(j_size):
            qi = Vn.getColumnVector(i)
            f_vec.axpy(x_coeff[i], qi)
            qi.destroy()

    Vn.assemble()

    t_basis_end = time()
    CPU_basis = t_basis_end - t_basis_start
    print(f"SOAR basis ({n} vectors) built in {CPU_basis:.2f} s")

    # --- Cleanup ---
    ksp.destroy()
    A0.destroy()
    A1.destroy()
    f_vec.destroy()
    u.destroy()
    if loading.freq_dep_order > 0:
        F.destroy()

    return Vn, CPU_basis, CPU_derivs, 0.0

### For WCAWE ###

def wcawe(simu: Simulation, f0: float, n: int, BSP: bool = True):
    """
    WCAWE procedure (Eq. 13 from Rumpler & Göransson, Proc. Mtgs. Acoust. 2017)
    adapted for model order reduction of the parametric system:

        [D1 + jk*D2 - k^2*D3] u(k) = F

    The system Z(x)*U(x) = F(x) is polynomial of degree 2 in frequency x.
    The l-th derivative of Z w.r.t. frequency, evaluated at expansion point x0:

        Z^(l)_{x0} = sum_{i=l}^{2} (2jπ/c)^i * i!/(i-l)! * x0^{i-l} * D_{i+1}

    All derivatives of order > 2 vanish (quadratic polynomial).

    WCAWE builds an orthonormal basis by moment matching with modified
    Gram-Schmidt orthogonalization and correction terms PQω to maintain
    well-conditioning. Each step requires one linear solve with Z^(0)
    (factorized once via MUMPS).

    Args:
        simu : Simulation object (provides operator.list_D and loading.F)
        f0   : Interpolation frequency in Hz
        n    : Number of basis vectors to generate
        BSP  : If True, apply Block-Structure-Preserving splitting (Eq. 22
               from Mariotti et al., JCP 2025). Doubles the reduced basis
               size but preserves block structure of coupled equations.

    Returns:
        Vn            : PETSc.Mat — orthonormal basis (N_dofs x n), or
                        (N_dofs x 2n) if BSP=True
        CPU_basis     : float — time to build the basis (seconds)
        CPU_derivs    : float — time to compute frequency derivatives (seconds)
        CPU_split     : float — time for BSP splitting (0 if BSP=False)
    """
    # --- Extract system components ---
    list_D = simu.operator.list_D
    degZ = len(list_D)  # = 3 for quadratic system
    loading = simu.loading
    if loading.freq_dep_order > 0:
        F = loading.F_at_freq(f0)
    else:
        F = loading.F_static
    F.assemble()

    # --- Compute frequency derivatives of Z (Eq. 19) ---
    # Z^(l)_{x0} = sum_{i=l}^{degZ-1} (2jπ/c)^i * i!/(i-l)! * x0^{i-l} * D_{i+1}
    t_deriv_start = time()
    list_Zj = []
    for l in range(degZ):
        Zl = None
        for i in range(l, degZ):
            coeff = (2j * np.pi / c0) ** i * factorial(i) / factorial(i - l) * f0 ** (i - l)
            if Zl is None:
                Zl = coeff * list_D[i]
            else:
                Zl = Zl + coeff * list_D[i]
        list_Zj.append(Zl)
    t_deriv_end = time()
    CPU_derivs = t_deriv_end - t_deriv_start
    print(f"WCAWE: frequency derivatives computed in {CPU_derivs:.4f} s")

    # --- Setup KSP solver (factorize Z^(0) once, reuse for all solves) ---
    ksp = PETSc.KSP().create()
    ksp.setOperators(list_Zj[0])
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    # --- Q matrix: stores Gram-Schmidt norms (diagonal) and projections (upper) ---
    Q_mat = PETSc.Mat().create()
    Q_mat.setSizes((n, n))
    Q_mat.setType("seqdense")
    Q_mat.setFromOptions()
    Q_mat.setUp()

    # --- Step 1: First basis vector  v1 = Z^(0)^{-1} F, normalized ---
    t_basis_start = time()
    v1 = list_Zj[0].createVecRight()
    ksp.solve(F, v1)
    norm_v1 = v1.norm(PETSc.NormType.NORM_2)
    Q_mat.setValue(0, 0, norm_v1)
    v1.scale(1.0 / norm_v1)

    N_dofs = v1.getSize()
    row_indices = list(range(N_dofs))

    # --- Initialize basis matrix Vn (dense, N_dofs x n) ---
    Vn = PETSc.Mat().create()
    Vn.setSizes((N_dofs, n))
    Vn.setType("seqdense")
    Vn.setFromOptions()
    Vn.setUp()
    Vn.setValues(row_indices, 0, v1.getArray(), PETSc.InsertMode.INSERT_VALUES)
    v1.destroy()

    # --- Precompute F derivative coefficients for frequency-dependent loading ---
    if loading.freq_dep_order > 0:
        F_coeffs = loading.F_deriv_coeffs(f0, min(loading.freq_dep_order, n))
        F_static = loading.F_static
        F_static.assemble()

    # --- Main WCAWE loop: build vectors v2, ..., vn (Eq. 13) ---
    for k in tqdm(range(2, n + 1), desc="WCAWE basis construction"):
        # First sum: F derivative contributions (non-zero for frequency-dependent loading)
        rhs_first_sum = list_Zj[0].createVecLeft()
        if loading.freq_dep_order > 0:
            for l in range(1, min(loading.freq_dep_order + 1, k)):
                if F_coeffs[l] != 0:
                    # F^(l)(f0) = F_coeffs[l] * F_static
                    F_l = F_static.copy()
                    F_l.scale(F_coeffs[l])

                    # Z^(0)^{-1} * F^(l)
                    temp = list_Zj[0].createVecRight()
                    ksp.solve(F_l, temp)
                    F_l.destroy()

                    # Correction term: P_Q_w scalar
                    P_q_1 = P_Q_w(Q_mat, k, l, 1)
                    P_q_1_col = P_q_1.getColumnVector(k - l - 1)
                    p_q_1_vals = P_q_1_col.getArray()
                    scalar_corr = p_q_1_vals[-1] if len(p_q_1_vals) > 0 else 0.0
                    rhs_first_sum.axpy(scalar_corr, temp)

                    temp.destroy()
                    P_q_1.destroy()
                    P_q_1_col.destroy()

        # Second sum: sum_{l=2}^{min(degZ-1, k-1)} Z^(l) * V[k-l] * PQ2(k,l) * e_{k-l}
        rhs_second_sum = list_Zj[0].createVecLeft()
        for l in range(2, min(degZ, k)):
            P_q_2 = P_Q_w(Q_mat, k, l, 2)
            P_q_2_col = P_q_2.getColumnVector(k - l - 1)

            row_is = PETSc.IS().createStride(N_dofs, first=0, step=1)
            col_is = PETSc.IS().createStride(k - l, first=0, step=1)
            Vn_sub = Vn.createSubMatrix(row_is, col_is)

            Zl_Vn_sub = list_Zj[l].matMult(Vn_sub)  # Z^(l) * V[k-l]
            contrib = Zl_Vn_sub.createVecLeft()
            Zl_Vn_sub.mult(P_q_2_col, contrib)       # Z^(l) * V[k-l] * PQ2 * e_{k-l}

            rhs_second_sum.axpy(1.0, contrib)

            row_is.destroy()
            col_is.destroy()
            P_q_2.destroy()
            P_q_2_col.destroy()
            Vn_sub.destroy()
            Zl_Vn_sub.destroy()
            contrib.destroy()

        # Z^(1) * v_{k-1} term
        rhs_Z1 = list_Zj[0].createVecLeft()
        vk_1 = Vn.getColumnVector(k - 2)
        list_Zj[1].mult(vk_1, rhs_Z1)
        vk_1.destroy()

        # RHS = first_sum - Z^(1)*v_{k-1} - second_sum
        rhs = rhs_first_sum.copy()
        rhs.axpy(-1.0, rhs_Z1)
        rhs.axpy(-1.0, rhs_second_sum)
        rhs_first_sum.destroy()
        rhs_Z1.destroy()
        rhs_second_sum.destroy()

        # Solve: Z^(0) * v_bar_k = rhs
        vk = list_Zj[0].createVecRight()
        ksp.solve(rhs, vk)
        rhs.destroy()

        # Modified Gram-Schmidt: store coefficients in Q and orthogonalize in one pass
        for i in range(k - 1):
            vi = Vn.getColumnVector(i)
            coeff = vk.dot(vi)
            Q_mat.setValue(i, k - 1, coeff)
            vk.axpy(-coeff, vi)
            vi.destroy()

        norm_vk = vk.norm(PETSc.NormType.NORM_2)

        # Deflation check
        if norm_vk < 1e-12:
            print(f"  WCAWE: deflation at step {k} (norm = {norm_vk:.2e})")
            Q_mat.setValue(k - 1, k - 1, 1.0)
            
            # Column stays zero (seqdense is zero-initialized)
        else:
            Q_mat.setValue(k - 1, k - 1, norm_vk)
            vk.scale(1.0 / norm_vk)
            Vn.setValues(row_indices, k - 1, vk.getArray(), PETSc.InsertMode.INSERT_VALUES)

        Q_mat.assemble()
        vk.destroy()

    Vn.assemble()
    t_basis_end = time()
    CPU_basis = t_basis_end - t_basis_start
    print(f"WCAWE basis ({n} vectors) built in {CPU_basis:.2f} s")

    # --- Cleanup solver ---
    ksp.destroy()
    for Zj in list_Zj:
        Zj.destroy()
    Q_mat.destroy()
    if loading.freq_dep_order > 0:
        F.destroy()

    # --- BSP splitting (optional) ---
    if BSP:
        Vn, CPU_split = split_basis_BSP(simu, Vn)
    else:
        CPU_split = 0.0

    return Vn, CPU_basis, CPU_derivs, CPU_split


def split_basis_BSP(simu: Simulation, Vn):
    """
    Block-Structure-Preserving (BSP) splitting of the WCAWE basis
    (Eq. 22 from Mariotti et al., JCP 2025).

    Splits each basis vector into its P-component (pressure) and
    Q-component (auxiliary variable), then orthonormalizes each
    sub-basis separately. The resulting block-diagonal basis:

        Ṽ = [Vp  0 ]
            [0   Vq]

    preserves the block structure of the original system matrices,
    improving convergence of the reduced model.

    Args:
        simu : Simulation object (provides mesh for DOF offset)
        Vn   : PETSc.Mat (N_dofs x n) — the WCAWE basis

    Returns:
        Vn_tilde  : PETSc.Mat (N_dofs x 2n) — BSP basis
        CPU_split : float — time for the splitting (seconds)
    """
    t_start = time()

    P, _ = simu.mesh.fonction_spaces()
    offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

    (N_dofs, n) = Vn.getSize()

    Vn_tilde = PETSc.Mat().create()
    Vn_tilde.setSizes((N_dofs, 2 * n))
    Vn_tilde.setType("seqdense")
    Vn_tilde.setFromOptions()
    Vn_tilde.setUp()

    for i in range(n):
        vi = Vn.getColumnVector(i)
        vp = vi[:offset].copy()
        vq = vi[offset:].copy()
        vi.destroy()

        # Gram-Schmidt orthogonalization against previous sub-vectors
        if i > 0:
            for ii in range(i):
                vii = Vn.getColumnVector(ii)
                vp_prev = vii[:offset]
                vq_prev = vii[offset:]
                vp -= np.vdot(vp_prev, vp) * vp_prev
                vq -= np.vdot(vq_prev, vq) * vq_prev
                vii.destroy()

        # Normalize
        norm_vp = np.linalg.norm(vp)
        norm_vq = np.linalg.norm(vq)
        if norm_vp > 1e-14:
            vp /= norm_vp
        if norm_vq > 1e-14:
            vq /= norm_vq

        Vn_tilde.setValues(list(range(offset)), i, vp, PETSc.InsertMode.INSERT_VALUES)
        Vn_tilde.setValues(list(range(offset, N_dofs)), n + i, vq, PETSc.InsertMode.INSERT_VALUES)

    Vn_tilde.assemble()
    Vn.destroy()

    t_end = time()
    CPU_split = t_end - t_start
    print(f"WCAWE: BSP splitting ({n} -> {2*n} vectors) in {CPU_split:.4f} s")

    return Vn_tilde, CPU_split


#### For rescaling ####

def ratio(A):
 
    print("computing cols_norms")
    cols_norms = column_norms(A)
    print("cols_norms computed")
    print("computing rows_norms")   
    rows_norms = row_norms(A)
    print("rows_norms computed")    
    ratio_row = max(rows_norms)/min(rows_norms)
    ratio_col = max(cols_norms)/min(cols_norms)
    if ratio_row > ratio_col:
        print('Ratio row is superior')
        return 'row', rows_norms, ratio_row
    else:
        print('Ratio col is superior')
        return 'col', cols_norms, ratio_col

def row_norms(A, ord=2):
    
    # Obtain the dimensions of the matrix
    m, n = A.getSize()

    # Initialise an array to store row norms
    list_row_norms = []

    # Calculate the norm of each line
    for i in range(m):
        row = A.getRow(i)[1]  # Retrieves non-zero values from the line
        row_norm = np.linalg.norm(row, ord=ord)
        list_row_norms.append(row_norm)
    return list_row_norms

def column_norms(A, ord=2):
    A_T = A.copy()
    A_T.transpose()
    return row_norms(A_T)

def rescaling_Z_entirely(Z, c_k, r_k, comm=PETSc.COMM_WORLD):
    norms_col = np.array(np.sqrt(c_k), dtype=np.float64)
    norms_row = np.array(np.sqrt(r_k), dtype=np.float64)

    n_col = len(norms_col)
    n_row = len(norms_row)

    # Créer une matrice carrée n×n
    Dc_inv = PETSc.Mat().create(comm=comm)
    Dc_inv.setType("aij")  # matrice creuse standard
    Dc_inv.setSizes([n_col, n_col])
    Dc_inv.setUp()

    Dr_inv = PETSc.Mat().create(comm=comm)
    Dr_inv.setType("aij")  # matrice creuse standard
    Dr_inv.setSizes([n_row, n_row])
    Dr_inv.setUp()

    # Définir la diagonale
    diag_vec_col = PETSc.Vec().createWithArray(1/norms_col, comm=comm)
    Dc_inv.setDiagonal(diag_vec_col, addv=PETSc.InsertMode.INSERT_VALUES)

    diag_vec_row = PETSc.Vec().createWithArray(1/norms_row, comm=comm)
    Dr_inv.setDiagonal(diag_vec_row, addv=PETSc.InsertMode.INSERT_VALUES)

    Dc_inv.assemble()
    Dr_inv.assemble()
    Z_rescaled = Dr_inv.matMult(Z)
    Z_rescaled = Z_rescaled.matMult(Dc_inv)
    return Z_rescaled, Dc_inv, Dr_inv

def check_convergence(r_k, c_k, atol):

    r_k = np.array(r_k)
    c_k = np.array(c_k)
    max_rows_diff = np.max(np.abs(r_k - 1))
    max_cols_diff = np.max(np.abs(c_k - 1))

    if max_rows_diff <= atol and max_cols_diff <= atol:
        return True
    else:
        return False

def algo_rescaling_Z(Z, comm=PETSc.COMM_WORLD, atol = 0.01):

    r_k = row_norms(Z, ord=np.inf)
    c_k = column_norms(Z, ord=np.inf)
    convergence = check_convergence(r_k, c_k, atol)
    
    if True: # Initialisation des matrices D1 et D2
        # Créer une matrice carrée n×n
        n_col = len(r_k)
        n_row = len(c_k)
        D1 = PETSc.Mat().create(comm=comm)
        D1.setType("aij")  # matrice creuse standard
        D1.setSizes([n_col, n_col])
        D1.setUp()

        D2 = PETSc.Mat().create(comm=comm)
        D2.setType("aij")  # matrice creuse standard
        D2.setSizes([n_row, n_row])
        D2.setUp()

        # Définir la diagonale
        diag_vec = PETSc.Vec().createWithArray([1 for _ in range(n_col)], comm=comm)
        D1.setDiagonal(diag_vec, addv=PETSc.InsertMode.INSERT_VALUES)
        D2.setDiagonal(diag_vec, addv=PETSc.InsertMode.INSERT_VALUES)

        D1.assemble()
        D2.assemble()
        diag_vec.destroy()

    if convergence:
        print('Matrix is already well scaled')
        return Z, D1, D2
    
    iteration = 0
    while not convergence and iteration < 100:
        
        Z, Dc_inv, Dr_inv = rescaling_Z_entirely(Z, c_k, r_k, comm=comm)
        D1 = D1.matMult(Dr_inv)
        D2 = D2.matMult(Dc_inv)
        
        r_k = row_norms(Z, ord=np.inf)
        c_k = column_norms(Z, ord=np.inf)
        convergence = check_convergence(r_k, c_k, atol)
        iteration += 1
    print(f'Number of iterations : {iteration}')
    return Z, D1, D2

### For computing metrics ###

def compute_analytical_radiation_factor(freqvec, radius):
    k_output = 2*np.pi*freqvec/c0
    Z_analytical = (1-2*special.jv(1,2*k_output*radius)/(2*k_output*radius) + 1j*2*special.struve(1,2*k_output*radius)/(2*k_output*radius)) #The impedance is divided by rho * c0, it becames the radiation coefficient
    return Z_analytical

def move_to_facet_quadrature(ufl_expr, mesh, submesh_info, scheme="default", degree=6):
    fdim = mesh.topology.dim - 1
    # Create submesh
    #bndry_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, fdim, sub_facets)
    bndry_mesh, entity_map = submesh_info[0], submesh_info[1][0]
    sub_top = entity_map.sub_topology
    assert isinstance(sub_top, dolfinx.mesh.Topology)
    sub_map = sub_top.index_map(entity_map.dim)
    indices = np.arange(sub_map.size_local + sub_map.num_ghosts, dtype=np.int32)
    parent_facets = entity_map.sub_topology_to_topology(indices, inverse=False)

    # Create quadrature space on submesh
    q_el = basix.ufl.quadrature_element(
        bndry_mesh.basix_cell(), ufl_expr.ufl_shape, scheme, degree
    )
    Q = dolfinx.fem.functionspace(bndry_mesh, q_el)

    # Compute where to evaluate expression per submesh cell
    integration_entities = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.exterior_facet, mesh.topology, parent_facets
    )

    compiled_expr = dolfinx.fem.Expression(ufl_expr, Q.element.interpolation_points, mesh.comm)

    # Evaluate expression
    q = dolfinx.fem.Function(Q)
    q.x.array[:] = compiled_expr.eval(
        mesh, integration_entities.reshape(-1, 2)
    ).reshape(-1)
    return q

### For data managing ###

def save_json(data, filename):
    with open(f"./raw_results/{filename}.json", 'w') as f:
        json.dump(data, f, indent=4)

def import_json(filename):
    with open(f"./raw_results/{filename}.json", 'r') as f:
        data = json.load(f)
    return data

def import_data(filename, tree = 'FOM/'):
    data = import_json(f'{tree}{filename}')
    Z_center_real = np.array(data['Z_center']['real'])
    freqvec = np.array(data['frequencies'])
    return freqvec, Z_center_real

### For plot ###

def plot_analytical_result_sigma(ax, freqvec, radius):
    Z_analytical = compute_analytical_radiation_factor(freqvec, radius)
    ax.plot(freqvec, Z_analytical.real, label = r'$\sigma_{ana}$', c = 'blue')
    ax.legend()

def harry_plotterv2(spaces, sols, str_values, show_edges = False):

    nb_graphs = len(spaces)

    if nb_graphs % 3 == 0 :
        rows = nb_graphs // 3
    else :
        rows = nb_graphs // 3 + 1 
    
    if nb_graphs <= 3 :
        cols = nb_graphs
    else : 
        cols = 3
    
    ii = 0
    plotter = pyvista.Plotter(shape=(rows, cols))

    for i in range(rows):
        for j in range(cols):
            space     = spaces[ii]
            sol       = sols[ii]
            str_value = str_values[ii]
            
            plotter.subplot(i, j)
            u_topology, u_cell_types, u_geometry = plot.vtk_mesh(space)
            u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
            u_grid.point_data[str_value] = sol.x.array.real
            u_grid.set_active_scalars(str_value)
            plotter.add_mesh(u_grid, show_edges=show_edges)
            plotter.view_xy()
            ii += 1
    
    if not pyvista.OFF_SCREEN:
        plotter.show()


def plot_pressure_slice_yz(space, sol, title='', clim=None):
    """Plot the pressure field on a (y, z) slice at x=0 using pyvista.

    Args:
        space : FunctionSpace (P space)
        sol   : Function — pressure solution to plot
        title : str — window title
        clim  : tuple (vmin, vmax) — color range, or None for auto
    """
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data['|p|'] = np.abs(sol.x.array)
    u_grid.set_active_scalars('|p|')

    sliced = u_grid.slice(normal='x', origin=(0, 0, 0))

    plotter = pyvista.Plotter()
    plotter.add_mesh(sliced, show_edges=False, cmap='viridis', clim=clim)
    plotter.view_yz()
    if title:
        plotter.add_title(title)
    if not pyvista.OFF_SCREEN:
        plotter.show()