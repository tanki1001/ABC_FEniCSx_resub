# Author Jorgen S. Dokken - Edited by Pierre Mariotti
# SPDX license identifier: MIT

import gmsh
from dolfinx.io import gmshio
from dolfinx import fem, plot
from mpi4py import MPI
import basix.ufl
import dolfinx.fem.petsc
import ufl
import numpy as np

from basix.ufl import element
from dolfinx.fem import form, assemble_scalar, Function, functionspace
from ufl import Measure

import pyvista

def tangential_proj(u, n):
    return (ufl.Identity(n.ufl_shape[0]) - ufl.outer(n, n)) * u

def move_to_facet_quadrature(ufl_expr, mesh, submesh_info, scheme="default", degree=40):
    fdim = mesh.topology.dim - 1
    # Create submesh
    #bndry_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, fdim, sub_facets)
    bndry_mesh, entity_map = submesh_info[0], submesh_info[1][0]
    #print(f'entity_map :{entity_map}')
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
    compiled_expr = dolfinx.fem.Expression(ufl_expr, Q.element.interpolation_points, comm=mesh.comm)

    # Evaluate expression
    q = dolfinx.fem.Function(Q)
    q.x.array[:] = compiled_expr.eval(
        mesh, integration_entities.reshape(-1, 2)
    ).reshape(-1)
    

    

    # --- 2) Champ "visu" en DG0 (Lagrange discontinu) pour plot ---
    Vvis = fem.functionspace(bndry_mesh, ("DG", 0))
    expr_vis = fem.Expression(ufl_expr, Vvis.element.interpolation_points, comm=mesh.comm)
    q_vis = fem.Function(Vvis)
    q_vis.x.array[:] = expr_vis.eval(mesh, integration_entities.reshape(-1, 2)).reshape(-1)




    harry_plotter(bndry_mesh, q_vis, 'kappa', show_edges = True)

    return q

def harry_plotter(mesh, sol, str_value, show_edges = True):

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(mesh)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.cell_data[str_value] = sol.x.array.real
    u_grid.set_active_scalars(str_value)
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=show_edges)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()

from geometries import spherical_domain, curvedcubic_domain, new_broken_cubic_domain, new_broken_cubic_domain_CAD, biSpherical_domain_CAD, cubicSpherical_domain_CAD, curved_cubic_domain_CAD, spherical_domain_CAD
#mesh_info, submesh_info = spherical_domain(side_box = 0.11, radius = 0.1, lc = 8e-3)
mesh_info, submesh_info = new_broken_cubic_domain_CAD(side_box=0.11, lc = 1.5e-2)
#mesh_info, submesh_info = cubicSpherical_domain_CAD()

domain = mesh_info[0]
submesh = submesh_info[0]

family = "Lagrange"
degP = 2
P1 = element(family, domain.basix_cell(), degP)
P  = functionspace(domain, P1)

ds  = Measure("ds", domain=domain, subdomain_data=mesh_info[2])

surfarea = assemble_scalar(form(1*ds(1)))
print(f'Surface area of the boundary: {surfarea}')
print(f'Expected surface area for sphere of radius 0.1: {1/4 * np.pi * 0.1**2}')
fx1 = Function(P)
fx1.interpolate(lambda x: 1/np.sqrt((x[0])**2 + (x[1])**2 + (x[2])**2))
n = ufl.FacetNormal(domain)
#kappa = n[0]
n = ufl.FacetNormal(domain)
kappa = ufl.div(n)
#kappa = move_to_facet_quadrature(kappa, domain, submesh_info)
gradt_c2 = (ufl.sqrt(ufl.grad(ufl.div(n))[0] ** 2 + ufl.grad(ufl.div(n))[1] ** 2 + ufl.grad(ufl.div(n))[2] ** 2))
#gradt_c2 = move_to_facet_quadrature(ufl.grad(ufl.div(n)), domain, submesh_info)
gradt_c2 = move_to_facet_quadrature(gradt_c2, domain, submesh_info)


kappa_vals = kappa.x.array
print(f'Mean curvature values: {kappa_vals}')
print(f'Max mean curvature value: {max(kappa_vals)}')
print(f'Expected mean curvature (constant) for sphere of radius 0.11: {2.0 / 0.11}')

# Plot the edge submesh (edges of Radiation_r surface)
edge_topology, edge_cell_types, edge_geometry = plot.vtk_mesh(edge_submesh)
edge_grid = pyvista.UnstructuredGrid(edge_topology, edge_cell_types, edge_geometry)
edge_plotter = pyvista.Plotter()
edge_plotter.add_mesh(edge_grid, color="red", line_width=5)
edge_plotter.add_title("Edges of Radiation_r (tag=3)")
edge_plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    edge_plotter.show()

