
# -*- coding: utf-8 -*-
"""
Génération d'une géométrie sphérique avec gmsh et import dans FEniCSx (dolfinx).

- Crée une sphère de rayon R centrée en (cx, cy, cz)
- Fixe une taille de maille globale lc
- Ajoute des groupes physiques:
    * dim=3, tag=1 : "volume" (l'intérieur de la sphère)
    * dim=2, tag=2 : "boundary" (la surface de la sphère)
- Génère le maillage 3D dans gmsh, puis convertit vers dolfinx via gmshio.model_to_mesh
- Sauvegarde optionnelle au format XDMF

Auteur: vous 🙂
"""

from mpi4py import MPI
import gmsh
import numpy as np

from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh
import ufl

import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI
import basix.ufl
import dolfinx.fem.petsc
import ufl
import numpy as np


def move_to_facet_quadrature(ufl_expr, mesh, sub_facets, scheme="default", degree=6):
    fdim = mesh.topology.dim - 1
    # Create submesh
    bndry_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, fdim, sub_facets)

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
    compiled_expr = dolfinx.fem.Expression(ufl_expr, Q.element.interpolation_points)

    # Evaluate expression
    q = dolfinx.fem.Function(Q)
    q.x.array[:] = compiled_expr.eval(
        mesh, integration_entities.reshape(-1, 2)
    ).reshape(-1)
    return q


def build_sphere_mesh_fenicsx(
    R: float = 1.0,
    center=(0.0, 0.0, 0.0),
    lc: float = 0.2,
    save_gmsh_msh: bool = False,
    save_xdmf: bool = True,
    model_name: str = "sphere"
):
    """
    Construit un maillage de sphère via gmsh, le convertit en dolfinx.Mesh, et (optionnel) sauvegarde.

    Paramètres
    ----------
    R : float
        Rayon de la sphère.
    center : tuple[float, float, float]
        Centre de la sphère (cx, cy, cz).
    lc : float
        Taille de maille cible globale (longueur caractéristique).
    save_gmsh_msh : bool
        Si True, sauvegarde le maillage gmsh (.msh).
    save_xdmf : bool
        Si True, sauvegarde le maillage et les tags dans un XDMF.
    model_name : str
        Nom du modèle gmsh.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        model = gmsh.model()
        gmsh.model.add(model_name)

        cx, cy, cz = center
        # Crée une sphère OCC. Selon votre version de gmsh, addSphere suffit (volume).
        # (Certaines versions historiques utilisaient "addBall"; on gère la compatibilité.)
        try:
            vol_tag = gmsh.model.occ.addSphere(cx, cy, cz, R)  # volume
        except Exception:
            # Fallback si votre installation a addBall
            vol_tag = gmsh.model.occ.addBall(cx, cy, cz, R)

        gmsh.model.occ.synchronize()

        # Taille de maille globale (simple et efficace pour un premier essai)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        # Pour des maillages plus fins proches des courbures :
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)

        # Groupes physiques
        vols = gmsh.model.getEntities(dim=3)
        surfs = gmsh.model.getEntities(dim=2)

        # 3D: volume = 1
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1)
        gmsh.model.setPhysicalName(3, 1, "volume")

        # 2D: surface = 2 (toute la surface sphérique)
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=2)
        gmsh.model.setPhysicalName(2, 2, "boundary")

        # Génération du maillage 3D
        gmsh.model.mesh.generate(3)
        model.mesh.setOrder(2)

    # Conversion gmsh -> dolfinx (le modèle doit rester vivant sur le rank 0)
    mesh_data = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3
    )
    print(mesh_data)
    mesh, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    if rank == 0:
        if save_gmsh_msh:
            gmsh.write(f"{model_name}.msh")
        # On peut finaliser gmsh après conversion
        gmsh.finalize()

    # Optionnel: sauvegarde au format XDMF (lisible par ParaView)
    

    # Mesures UFL avec subdomain_data pour faciliter l’assemblage
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # Petit résumé sur le rank 0
    if rank == 0:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        num_verts = mesh.geometry.x.shape[0]
        print(f"[Résumé] Élts: {num_cells}, Sommets: {num_verts}, R={R}, lc={lc}")
        print("Tags disponibles:")
        print(" - Volume (dim=3), tag=1 → 'volume'")
        print(" - Surface (dim=2), tag=2 → 'boundary'")

    return mesh, cell_tags, facet_tags, dx, ds



domain, cell_tags, facet_tags, dx, ds = build_sphere_mesh_fenicsx(
    R=1.0,
    center=(0.0, 0.0, 0.0),
    lc=0.05,
    save_gmsh_msh=False,
    save_xdmf=True,
    model_name="sphere"
)



n = ufl.FacetNormal(domain)


kappa = ufl.div(n)


domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
exterior_boundaries = dolfinx.mesh.exterior_facet_indices(domain.topology)


kappa = move_to_facet_quadrature(kappa, domain, exterior_boundaries)

kappa_vals = kappa.x.array
print(f'Mean curvature values on the membrane facets: {kappa_vals}')
print(f'Expected mean curvature (constant) for sphere of radius 1: {2.0 / 1}')


