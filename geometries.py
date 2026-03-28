from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio
import dolfinx.mesh as msh
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc

def cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "Cubic"):

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p6 = gmsh.model.geo.addPoint(side_box, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box, side_box, side_box, lc)
    p8 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)

    p9 = gmsh.model.geo.addPoint(side_box, radius, 0, lc)
    p10 = gmsh.model.geo.addPoint(side_box, 0, radius, lc)

    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p9)
    l4 = gmsh.model.geo.addLine(p9, p2)
    l5 = gmsh.model.geo.addLine(p2, p1)

    l6 = gmsh.model.geo.addLine(p5, p6)
    l7 = gmsh.model.geo.addLine(p6, p7)
    l8 = gmsh.model.geo.addLine(p7, p8)
    l9 = gmsh.model.geo.addLine(p8, p5)

    l10 = gmsh.model.geo.addLine(p2, p10)
    l11 = gmsh.model.geo.addLine(p10, p6)
    l12 = gmsh.model.geo.addLine(p3, p7)
    l13 = gmsh.model.geo.addLine(p4, p8)
    l14 = gmsh.model.geo.addLine(p1, p5)

    # definition of the quarter circle
    c15 = gmsh.model.geo.addCircleArc(p9, p2, p10)

    # Curve loops
    cl1 = [l1, l2, l3, l4, l5]
    cl2 = [l6, l7, l8, l9]
    cl3 = [-l3, l12, -l7, -l11, -c15]
    cl4 = [c15, -l10, -l4]
    cl5 = [-l2, l13, -l8, -l12]
    cl6 = [-l5, l10, l11, -l6, -l14]
    cl7 = [-l1, l14, -l9, -l13]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl7)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(1, [l12, -l7, -l6, -l14, l1, l2], tag=111)
    gmsh.model.addPhysicalGroup(2, [s4], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s7, s5, s2], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)
    

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    ridge_tags = mesh_data.ridge_tags   # dict lookup -> ça marche
    # <--- pas officiel, il faut les récupérer
    print(f'ridge_tags.indices : {ridge_tags.indices}')  # entités locales de dimension 1
    print(f'ridge_tags.values : {ridge_tags.values}')
    print(f'facet_tags.indices : {facet_tags.indices}')  # entités locales de dimension 1
    print(f'facet_tags.values : {facet_tags.values}')

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [side_box, 0, 0]

    

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]
    

    #mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    mesh_info = [final_mesh, cell_tags, facet_tags, xref, ridge_tags] # this line is new
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def curvedcubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "rounded_cubic"):

    rc = side_box / 10

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box - rc, 0, lc)
    p4 = gmsh.model.geo.addPoint(side_box - rc, side_box, 0, lc)
    p5 = gmsh.model.geo.addPoint(0, side_box, 0, lc)
    p6 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box - rc, 0, side_box, lc)
    p8 = gmsh.model.geo.addPoint(side_box - rc, side_box - rc, side_box, lc)
    p9 = gmsh.model.geo.addPoint(0, side_box - rc, side_box, lc)
    p10 = gmsh.model.geo.addPoint(side_box, 0, side_box - rc, lc)
    p11 = gmsh.model.geo.addPoint(side_box, side_box - rc, side_box - rc, lc)
    p12 = gmsh.model.geo.addPoint(side_box - rc, side_box, side_box - rc, lc)
    p13 = gmsh.model.geo.addPoint(0, side_box, side_box - rc, lc)
    p14 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    p15 = gmsh.model.geo.addPoint(0, 0, radius, lc)

    # Defining points for circles
    pc4 = gmsh.model.geo.addPoint(side_box - rc, side_box - rc, 0, lc)
    pc20_14_23 = gmsh.model.geo.addPoint(side_box - rc, side_box - rc, side_box - rc, lc)
    pc18 = gmsh.model.geo.addPoint(0, side_box - rc, side_box - rc, lc)
    pc10 = gmsh.model.geo.addPoint(side_box - rc, 0, side_box - rc, lc)

    
    # Definition of the lines
    c0 = gmsh.model.geo.addCircleArc(p14, p1, p15)
    l1 = gmsh.model.geo.addLine(p1, p14)
    l2 = gmsh.model.geo.addLine(p14, p2)
    l3 = gmsh.model.geo.addLine(p2, p3)
    c4 = gmsh.model.geo.addCircleArc(p3, pc4, p4)
    l5 = gmsh.model.geo.addLine(p4, p5)
    l6 = gmsh.model.geo.addLine(p5, p1)
    l7 = gmsh.model.geo.addLine(p1, p15)
    l8 = gmsh.model.geo.addLine(p15, p6)
    l9 = gmsh.model.geo.addLine(p6, p7)
    c10 = gmsh.model.geo.addCircleArc(p7, pc10, p10)
    l11 = gmsh.model.geo.addLine(p10, p2)
    l12 = gmsh.model.geo.addLine(p3, p11)
    l13 = gmsh.model.geo.addLine(p11, p10)
    c14 = gmsh.model.geo.addCircleArc(p11, pc20_14_23, p8)
    l15 = gmsh.model.geo.addLine(p7, p8)
    l16 = gmsh.model.geo.addLine(p8, p9)
    l17 = gmsh.model.geo.addLine(p9, p6)
    c18 = gmsh.model.geo.addCircleArc(p9, pc18, p13)
    l19 = gmsh.model.geo.addLine(p13, p12)
    c20 = gmsh.model.geo.addCircleArc(p12, pc20_14_23, p11)
    l21 = gmsh.model.geo.addLine(p12, p4)
    l22 = gmsh.model.geo.addLine(p5, p13)
    c23 = gmsh.model.geo.addCircleArc(p8, pc20_14_23, p12)

    # Curve loops
    cl1 = [l1, c0, -l7]
    cl2 = [l2, -l11, -c10, -l9, -l8, -c0]
    cl3 = [l3, l12, l13, l11]
    cl4 = [c4, -l21, c20, -l12]
    cl5 = [l5, l22, l19, l21]
    cl6 = [l8, -l17, c18, -l22, l6, l7]
    cl7 = [c23, -l19, -c18, -l16]
    cl8 = [l17, l9, l15, l16]
    cl9 = [c14, -l15, c10, -l13]
    cl10 = [-c20, -c23, -c14]
    cl11 = [-l2, -l1, -l6, -l5, -c4, -l3]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl7)])
    s8 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl8)])
    s9 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl9)])
    s10 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl10)])
    s11 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl11)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7, s8, s9, s10, s11])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s3, s4, s5, s7, s10, s9, s8], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique


    # Adaptation à la courbure CAD
    #gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)    #-> impossible to use
    #gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 6) #-> impossible to use

    # Generation of the 3D mesh
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)
    #model.mesh.setOrder(2)

    

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def spherical_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "spherical"):

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p3 = gmsh.model.geo.addPoint(0, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, radius, lc)
    p6 = gmsh.model.geo.addPoint(0, radius, 0, lc)


    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p5)
    l2 = gmsh.model.geo.addLine(p5, p2)
    c3 = gmsh.model.geo.addCircleArc(p2, p1, p3)
    l4 = gmsh.model.geo.addLine(p3, p6)
    l5 = gmsh.model.geo.addLine(p6, p1)

    c6 = gmsh.model.geo.addCircleArc(p5, p1, p6)

    c7 = gmsh.model.geo.addCircleArc(p2, p1, p4)
    l8 = gmsh.model.geo.addLine(p4, p1)

    c9 = gmsh.model.geo.addCircleArc(p4, p1, p3)


    # Curve loops
    cl1 = gmsh.model.geo.addCurveLoop([l1, c6, l5])
    cl2 = gmsh.model.geo.addCurveLoop([l2, c3, l4, -c6])
    cl3 = gmsh.model.geo.addCurveLoop([-l1, -l8, -c7, -l2])
    cl4 = gmsh.model.geo.addCurveLoop([-c9, l8, -l5, -l4])
    cl5 = gmsh.model.geo.addCurveLoop([c9, -c3, c7])


    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    s5 = gmsh.model.geo.addSurfaceFilling([cl5])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s4, s3])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s5], tag=3)  # Radiation_r
    gmsh.model.addPhysicalGroup(1, [c7, c3, c9], tag=10)  # Edges of Radiation_r
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique


    # Generation of the 3D mesh
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    ridge_tags = mesh_data.ridge_tags   # dict lookup -> ça marche
    print(f'ridge_tags.indices : {ridge_tags.indices}')  # entités locales de dimension 1
    print(f'ridge_tags.values : {ridge_tags.values}')

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    entity_maps_mesh = [entity_map]

    # Create edge submesh from the edges of Radiation_r (physical group tag=10)
    edim = tdim - 2  # dimension 1
    edge_submesh, edge_entity_map = msh.create_submesh(final_mesh, edim, ridge_tags.find(10))[0:2]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref, ridge_tags]
    submesh_info = [submesh, entity_maps_mesh]

    #return mesh_info, submesh_info, edge_submesh
    return mesh_info, submesh_info

def ellipsoidal_domain(side_box=0.11, radius=0.1, lc=8e-3, model_name="ellipsoid"):
    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p3 = gmsh.model.geo.addPoint(0, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(2*side_box, 0, 0, lc)
    p5 = gmsh.model.geo.addPoint(0, 0, radius, lc)
    p6 = gmsh.model.geo.addPoint(0, radius, 0, lc)

    # Definition of the lines and arcs
    l1 = gmsh.model.geo.addLine(p1, p5)
    l2 = gmsh.model.geo.addLine(p5, p2)
    c3 = gmsh.model.geo.addCircleArc(p2, p1, p3)  # Part of cube's edge
    l4 = gmsh.model.geo.addLine(p3, p6)
    l5 = gmsh.model.geo.addLine(p6, p1)
    c6 = gmsh.model.geo.addCircleArc(p5, p1, p6)  # Circular arc for surface 1

    # Elliptical arcs replacing spherical arcs
    # Elliptical arc from p2 (0,0,side_box) to p4 (side_box,0,0) in x-z plane
    e7 = gmsh.model.geo.addEllipseArc(p2, p1, p4, p4)
    # Elliptical arc from p4 (side_box,0,0) to p3 (0,side_box,0) in x-y plane
    e9 = gmsh.model.geo.addEllipseArc(p4, p1, p4, p3)
    l8 = gmsh.model.geo.addLine(p4, p1)

    # Curve loops
    cl1 = gmsh.model.geo.addCurveLoop([l1, c6, l5])
    cl2 = gmsh.model.geo.addCurveLoop([l2, c3, l4, -c6])
    cl3 = gmsh.model.geo.addCurveLoop([-l1, -l8, -e7, -l2])
    cl4 = gmsh.model.geo.addCurveLoop([-e9, l8, -l5, -l4])
    cl5 = gmsh.model.geo.addCurveLoop([e9, -c3, e7])

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    s5 = gmsh.model.geo.addSurfaceFilling([cl5])  # Ellipsoidal surface

    # Surface loop and volume
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s4, s3])
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Surface quart de disque
    gmsh.model.addPhysicalGroup(2, [s5], tag=3)  # Surface ellipsoïdale
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)   # Volume

    # Mesh generation
    gmsh.model.mesh.generate(3)
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    gmsh.finalize()

    # Remaining code for submesh and entity maps...
    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    xref = [0, 0, 0]
    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info




def half_cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "half_cubic"):

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box/2, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box/2, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p6 = gmsh.model.geo.addPoint(side_box/2, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box/2, side_box, side_box, lc)
    p8 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)

    p9 = gmsh.model.geo.addPoint(side_box/2, radius, 0, lc)
    p10 = gmsh.model.geo.addPoint(side_box/2, 0, radius, lc)

    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p9)
    l4 = gmsh.model.geo.addLine(p9, p2)
    l5 = gmsh.model.geo.addLine(p2, p1)

    l6 = gmsh.model.geo.addLine(p5, p6)
    l7 = gmsh.model.geo.addLine(p6, p7)
    l8 = gmsh.model.geo.addLine(p7, p8)
    l9 = gmsh.model.geo.addLine(p8, p5)

    l10 = gmsh.model.geo.addLine(p2, p10)
    l11 = gmsh.model.geo.addLine(p10, p6)
    l12 = gmsh.model.geo.addLine(p3, p7)
    l13 = gmsh.model.geo.addLine(p4, p8)
    l14 = gmsh.model.geo.addLine(p1, p5)

    # definition of the quarter circle
    c15 = gmsh.model.geo.addCircleArc(p9, p2, p10)

    # Curve loops
    cl1 = [l1, l2, l3, l4, l5]
    cl2 = [l6, l7, l8, l9]
    cl3 = [-l3, l12, -l7, -l11, -c15]
    cl4 = [c15, -l10, -l4]
    cl5 = [-l2, l13, -l8, -l12]
    cl6 = [-l5, l10, l11, -l6, -l14]
    cl7 = [-l1, l14, -l9, -l13]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl7)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s4], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s7, s5, s2], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [side_box/2, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def half_curvedcubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "rounded_cubic"):

    rc = side_box / 10

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

   # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box/2 - rc, 0, lc)
    p4 = gmsh.model.geo.addPoint(side_box - rc, side_box/2, 0, lc)
    p5 = gmsh.model.geo.addPoint(0, side_box/2, 0, lc)
    p6 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box - rc, 0, side_box, lc)
    p8 = gmsh.model.geo.addPoint(side_box - rc, side_box/2 - rc, side_box, lc)
    p9 = gmsh.model.geo.addPoint(0, side_box/2 - rc, side_box, lc)
    p10 = gmsh.model.geo.addPoint(side_box, 0, side_box - rc, lc)
    p11 = gmsh.model.geo.addPoint(side_box, side_box/2 - rc, side_box - rc, lc)
    p12 = gmsh.model.geo.addPoint(side_box - rc, side_box/2, side_box - rc, lc)
    p13 = gmsh.model.geo.addPoint(0, side_box/2, side_box - rc, lc)
    p14 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    p15 = gmsh.model.geo.addPoint(0, 0, radius, lc)

    # Defining points for circles
    pc4 = gmsh.model.geo.addPoint(side_box - rc, side_box/2 - rc, 0, lc)
    pc20_14_23 = gmsh.model.geo.addPoint(side_box - rc, side_box/2 - rc, side_box - rc, lc)
    pc18 = gmsh.model.geo.addPoint(0, side_box/2 - rc, side_box - rc, lc)
    pc10 = gmsh.model.geo.addPoint(side_box - rc, 0, side_box - rc, lc)

    
    # Definition of the lines
    c0 = gmsh.model.geo.addCircleArc(p14, p1, p15)
    l1 = gmsh.model.geo.addLine(p1, p14)
    l2 = gmsh.model.geo.addLine(p14, p2)
    l3 = gmsh.model.geo.addLine(p2, p3)
    c4 = gmsh.model.geo.addCircleArc(p3, pc4, p4)
    l5 = gmsh.model.geo.addLine(p4, p5)
    l6 = gmsh.model.geo.addLine(p5, p1)
    l7 = gmsh.model.geo.addLine(p1, p15)
    l8 = gmsh.model.geo.addLine(p15, p6)
    l9 = gmsh.model.geo.addLine(p6, p7)
    c10 = gmsh.model.geo.addCircleArc(p7, pc10, p10)
    l11 = gmsh.model.geo.addLine(p10, p2)
    l12 = gmsh.model.geo.addLine(p3, p11)
    l13 = gmsh.model.geo.addLine(p11, p10)
    c14 = gmsh.model.geo.addCircleArc(p11, pc20_14_23, p8)
    l15 = gmsh.model.geo.addLine(p7, p8)
    l16 = gmsh.model.geo.addLine(p8, p9)
    l17 = gmsh.model.geo.addLine(p9, p6)
    c18 = gmsh.model.geo.addCircleArc(p9, pc18, p13)
    l19 = gmsh.model.geo.addLine(p13, p12)
    c20 = gmsh.model.geo.addCircleArc(p12, pc20_14_23, p11)
    l21 = gmsh.model.geo.addLine(p12, p4)
    l22 = gmsh.model.geo.addLine(p5, p13)
    c23 = gmsh.model.geo.addCircleArc(p8, pc20_14_23, p12)
    

    # Curve loops
    cl1 = [l1, c0, -l7]
    cl2 = [l2, -l11, -c10, -l9, -l8, -c0]
    cl3 = [l3, l12, l13, l11]
    cl4 = [c4, -l21, c20, -l12]
    cl5 = [l5, l22, l19, l21]
    cl6 = [l8, -l17, c18, -l22, l6, l7]
    cl7 = [c23, -l19, -c18, -l16]
    cl8 = [l17, l9, l15, l16]
    cl9 = [c14, -l15, c10, -l13]
    cl10 = [-c20, -c23, -c14]
    cl11 = [-l2, -l1, -l6, -l5, -c4, -l3]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl7)])
    s8 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl8)])
    s9 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl9)])
    s10 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl10)])
    s11 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl11)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7, s8, s9, s10, s11])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s3, s4, s5, s7, s10, s9, s8], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    #gmsh.option.setNumber("Mesh.ElementOrder", 2)
    #gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def broken_cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):
    #Make an eighth of the cubic domain of a sound box
    
    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    model_name = "example"
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p6 = gmsh.model.geo.addPoint(side_box, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box, side_box, side_box, lc)
    p8 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)

    p9 = gmsh.model.geo.addPoint(side_box, radius, 0, lc)
    p10 = gmsh.model.geo.addPoint(side_box, 0, radius, lc)

    p11 = gmsh.model.geo.addPoint(side_box/2, side_box, side_box/2, lc)
    p12 = gmsh.model.geo.addPoint(side_box/2, 0, side_box/2, lc)

    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p9)
    l4 = gmsh.model.geo.addLine(p9, p2)
    l5 = gmsh.model.geo.addLine(p2, p1)

    l6 = gmsh.model.geo.addLine(p5, p6)
    l7 = gmsh.model.geo.addLine(p6, p7)
    l8 = gmsh.model.geo.addLine(p7, p8)
    l9 = gmsh.model.geo.addLine(p8, p5)

    l10 = gmsh.model.geo.addLine(p2, p10)
    l11 = gmsh.model.geo.addLine(p10, p6)
    l12 = gmsh.model.geo.addLine(p3, p7)
    l13 = gmsh.model.geo.addLine(p4, p11)
    l14 = gmsh.model.geo.addLine(p11, p8)
    l16 = gmsh.model.geo.addLine(p1, p12)
    l17 = gmsh.model.geo.addLine(p12, p5)
    l18 = gmsh.model.geo.addLine(p11, p12)
    # definition of the quarter circle
    c15 = gmsh.model.geo.addCircleArc(p9, p2, p10)

    # Curve loops
    cl1 = [l1, l2, l3, l4, l5]
    cl2 = [l6, l7, l8, l9]
    cl3 = [-l3, l12, -l7, -l11, -c15]
    cl4 = [c15, -l10, -l4]
    cl5 = [-l2, l13, l14, -l8, -l12]
    cl6 = [-l5, l10, l11, -l6, -l17, -l16]
    cl7 = [-l1, l16, -l18, -l13]
    cl8 = [l18, l17, -l9, -l14]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl7)])
    s8 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl8)])
    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7, s8])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s4], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s8, s7, s5, s2], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write("Broken_cubic.msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [side_box, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info
    
def new_broken_cubic_domain_byHand(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):
    #Make an eighth of the cubic domain of a sound box
    side_box = 0.11
    radius = 0.1
    rc = side_box / 10
    side = side_box


    # ────────────────────────── initialisation ──────────────────────────
    gmsh.initialize()
    gmsh.model.add("new_broken_python")


    t8 = np.tan(np.pi/8)
    s4 = np.sin(np.pi/4)
    c4 = np.cos(np.pi/4)
    s8 = np.sin(np.pi/8)

    addPt   = gmsh.model.geo.addPoint
    addLine = gmsh.model.geo.addLine
    addCirc = gmsh.model.geo.addCircleArc

    # ───────────────────────────── Points p1…p22 ─────────────────────────
    addPt(0,      0,      0,      lc, tag=1)
    addPt(side,   0,      0,      lc, tag=2)
    addPt(side-rc,0,      side,   lc, tag=3)
    addPt(0,      0,      side,   lc, tag=4)

    addPt(0,      side-rc*t8,      side,           lc, tag=5)
    addPt(side-rc,side-rc*t8,      side,           lc, tag=6)

    addPt(0,      side-rc*t8+rc*c4,side-rc*(1+s4), lc, tag=7)
    addPt(side-rc,side-rc*t8+rc*c4,side-rc*(1+s4), lc, tag=8)

    addPt(side-rc,side/2+rc*(1-c4),side/2+rc*(1-s4), lc, tag=9)
    addPt(0,       side/2+rc*(1-c4),side/2+rc*(1-s4), lc, tag=10)

    addPt(0,          side/2,          side/2-rc*s8, lc, tag=11)
    addPt(side-rc,    side/2,          side/2-rc*s8, lc, tag=12)

    addPt(side-rc,    side/2,          0,            lc, tag=13)
    addPt(0,          side/2,          0,            lc, tag=14)

    addPt(side,       side/2-rc,       0,            lc, tag=15)
    addPt(side,       side/2-rc,       side/2-rc*s8, lc, tag=16)

    addPt(side,       side/2+rc*(1-2*c4),
                    side/2-rc*s8+2*rc*s4,           lc, tag=17)

    addPt(side,       side-rc*t8+rc*s4,
                    side-rc*(1-s4),                 lc, tag=18)

    addPt(0,          0,               radius,       lc, tag=19)
    addPt(radius,     0,               0,            lc, tag=20)

    addPt(side, 0,            side-rc,              lc, tag=21)
    addPt(side, side-rc*t8,   side-rc,              lc, tag=22)

    # ───────── centres d’arc (23…27 puis 233 etc. gardés identiques) ────
    addPt(0,          side-rc*t8, side-rc,          lc, tag=23)
    addPt(side,       side-rc*t8, side-rc,          lc, tag=233)

    addPt(side-rc,    side/2-rc,  0,                lc, tag=24)
    addPt(side-rc,    side/2-rc,  side/2-rc*s8,     lc, tag=244)

    addPt(0,          side/2+rc,  side/2-rc*s8,     lc, tag=25)
    addPt(side-rc,    side/2+rc,  side/2-rc*s8,     lc, tag=255)
    addPt(side,       side/2+rc,  side/2-rc*s8,     lc, tag=2555)

    addPt(side-rc,    0,          side-rc,          lc, tag=26)
    addPt(side-rc,    side-rc,    side-rc,          lc, tag=266)

    addPt(side-3*rc/2, side/2-rc/2, side/2+3*rc/2,  lc, tag=27)
    addPt(side-3*rc/2, side-3*rc/2, side-3*rc/2,    lc, tag=277)

    # ─────────────────────────── courbes 1…35 ───────────────────────────
    addLine(1 ,20, tag=1)
    addLine(20, 2, tag=2)
    addLine(2 ,21, tag=3)
    addLine(3 , 4, tag=4)
    addLine(4 ,19, tag=5)
    addLine(19, 1, tag=6)
    addCirc(19,1 ,20, tag=7)

    addLine(3 ,6 , tag=8)
    addLine(6 ,5 , tag=9)
    addLine(5 ,4 , tag=10)
    addCirc(5 ,23,7 , tag=11)

    addLine(7 ,8 , tag=13)
    addLine(8 ,9 , tag=14)
    addLine(9 ,10, tag=15)
    addLine(10,7 , tag=16)
    addCirc(6 ,233,8 , tag=17)
    addCirc(9 ,27 ,17, tag=18)

    addLine(17,22, tag=19)
    addCirc(17,2555,16, tag=20)
    addCirc(16,244 ,12, tag=21)
    addCirc(12,255 ,9 , tag=22)
    addLine(12,11, tag=23)
    addCirc(11,25 ,10, tag=24)

    addLine(11,14, tag=25)
    addLine(14,13, tag=26)
    addLine(13,12, tag=27)
    addCirc(13,24 ,15, tag=28)
    addLine(15,16, tag=29)
    addLine(15, 2, tag=30)
    addLine(1 ,14, tag=31)

    addCirc(22,266,6 , tag=32)
    addCirc(8 ,277,22, tag=33)
    addCirc(3 ,26 ,21, tag=35)

    # ────────────────────────── synchronisation ─────────────────────────
    gmsh.model.geo.synchronize()

    # ─────────────────────────—— visualisation rapide (optionnel) ───────
    # gmsh.fltk.run()

    # ────────────────────────── sauvegarde / maillage ───────────────────
    gmsh.write("new_broken_from_python.geo")   # script équivalent
    # gmsh.model.mesh.generate(3)
    # gmsh.write("new_broken_from_python.msh")

    gmsh.finalize()

def new_broken_cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):
    
    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)
    

    gmsh.option.setNumber("General.Terminal", 1)   # verbosité console
    gmsh.merge("nastran_mesh_new_broken.nas")                       # ← votre STL

    # (1)   Classifier les triangles en patches homogènes
    angle = 20       # seuil (°) entre triangles d'une même face
    force_param = True          # impose une surface paramétrable
    include_boundary = True     # recolle aussi les arêtes frontières
    curve_angle = 180           # subdivision de cônes s'ils dépassent
    #gmsh.model.mesh.classifyFaces(angle*math.pi/180, force_param,
    #                              include_boundary, curve_angle*math.pi/180)
    gmsh.model.mesh.classifySurfaces(angle*np.pi/180, force_param,
                                include_boundary, curve_angle*np.pi/180)

    # (2)   Générer la géométrie B-Rep à partir du maillage
    #gmsh.model.mesh.createGeometry()
    # après classifySurfaces()
    
    s = gmsh.model.getEntities(dim=2)       # surfaces existantes
    surfTag = s[0][1]                       # ici la face plane fusionnée
    eleType, elemTags, nodeTags = gmsh.model.mesh.getElements(2, surfTag)
    triangles = elemTags[0]                 # 1er (et seul) bloc : triangles

    # (3)   Récupérer tout le volume et ses faces
    volumes   = gmsh.model.getEntities(dim=3)
    surfaces  = gmsh.model.getEntities(dim=2)
    print(f'surfaces: {surfaces}')

    # (4)   Physical groups : 1 pour le volume, un par face
    vol_tag = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.setPhysicalName(3, vol_tag, "SOLID")

    for i, (dim, tag) in enumerate(surfaces, start=1):
            if tag == 19:
                pg = gmsh.model.addPhysicalGroup(dim, [tag], tag=3)
                gmsh.model.setPhysicalName(dim, pg, f"SURF_{i}")
                print(f'i, tag : {i},{tag}')
            else :
                pg = gmsh.model.addPhysicalGroup(dim, [tag], tag=i+3000)
                gmsh.model.setPhysicalName(dim, pg, f"SURF_{i}")
                print(f'i, tag : {i},{tag}')
    # ------------------------------------------------------------------
    #  Sélection de la surface plane située sur x=0
    # ------------------------------------------------------------------
    # on prend la première surface dont le barycentre a |x| < 1e-4



    # ------------------------------------------------------------------
    #  Récupération des triangles et de leurs nœuds
    # ------------------------------------------------------------------
    etypes, elemTags, nodeTags = gmsh.model.mesh.getElements(2, 17)
    tri_tags   = elemTags[0]          # liste des IDs d'élément
    tri_nodes  = nodeTags[0]          # nœuds concaténés  (nElem*3)

    # table id noeud -> coord
    nodeIds, xyz, _ = gmsh.model.mesh.getNodes()
    coords = dict(zip(nodeIds, xyz.reshape(-1, 3)))

    # ------------------------------------------------------------------
    #  Critère quart-de-disque
    # ------------------------------------------------------------------
    xc, yc, zc = 0.0, 0.0, 0.0        # centre
    R          = radius                # rayon
    tri_QD, tri_rest = [], []

    for k, etag in enumerate(tri_tags):
        n1, n2, n3 = tri_nodes[3*k:3*k+3]      # id des 3 sommets
        x = (coords[n1][0] + coords[n2][0] + coords[n3][0]) / 3
        y = (coords[n1][1] + coords[n2][1] + coords[n3][1]) / 3
        z = (coords[n1][2] + coords[n2][2] + coords[n3][2]) / 3

        # plan x≈0 déjà garanti ; on teste le quart de disque
        dy, dz = y - yc, z - zc
        if (dy >= 0) and (dz >= 0) and (dy*dy + dz*dz <= R*R + 1e-10):
            tri_QD.append(etag)
        else:
            tri_rest.append(etag)

    print(f"quart-de-disque : {len(tri_QD)} tri.  |  reste : {len(tri_rest)} tri.")

    # ------------------------------------------------------------------
    # 1) créer une nouvelle surface discrète pour le quart de disque
    # ------------------------------------------------------------------
    planeTag = 17
    # 1) nouvelle surface discrète
    surf_qd = gmsh.model.addDiscreteEntity(2)
    triType = gmsh.model.mesh.getElementType("triangle", 1)

    # 2) connectivité aplatie des 177 triangles sélectionnés
    tri_nodes_qd = []
    for k, etag in enumerate(tri_tags):
        if etag in tri_QD:
            tri_nodes_qd.extend(tri_nodes[3*k:3*k+3])

    # 3) ajout dans la nouvelle surface (IDs auto ou ré-utilisés)
    elem_ids = tri_QD                     # ou  list(range(1, len(tri_QD)+1))

    gmsh.model.mesh.addElementsByType(
            surf_qd,                      # surface discrète (entityTag)
            triType,                      # type 'triangle' P1
            elem_ids,                     # ← elementTags
            tri_nodes_qd)                 # ← nodeTags (aplati)
    # 4) on retire ces triangles de la surface d’origine (sinon doublons)
    #gmsh.model.mesh.removeElementsByType(2, planeTag, triType, tri_QD)

    # 5) Physical groups (entités géométriques ✓)
    pg_qd   = gmsh.model.addPhysicalGroup(2, [surf_qd], tag=1)
    gmsh.model.setPhysicalName(2, pg_qd, "SURF_QD")
    #print(f'surf_qd : {surf_qd}')

    pg_rest = gmsh.model.addPhysicalGroup(2, [planeTag], tag=2002)
    gmsh.model.setPhysicalName(2, pg_rest, "SURF_REST")

    gmsh.model.geo.synchronize()

    # (5)   Maillage 3-D + export
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.002)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.002)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 8e-3)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 8e-3)
    gmsh.model.mesh.generate(3)
    model.mesh.setOrder(2)
    gmsh.write("nastran_comsol_mesh_to_gmsh.msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    mesh_data = gmshio.model_to_mesh(model, comm, model_rank)
    final_mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    # Close gmsh

    gmsh.finalize()
    print("✔  Maillage écrit : nastran_comsol_mesh_to_gmsh.msh")


    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info





def new_broken_cubic_domain_CAD(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):

    gmsh.initialize()
    
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("from_comsol")
    if side_box == 0.11:
        case = "small"
    elif side_box == 0.40:
        case = "large"
    else:
        raise ValueError("side_box must be either 0.11 or 0.40")

    gmsh.model.occ.importShapes(f"{case}_object_new_broken_cubic_case.step")
    gmsh.model.occ.synchronize()
    
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)

    
    volumes = gmsh.model.getEntities(dim=3)
    print(f'volumes: {volumes}')
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15], tag=3)


    
    # Adaptation à la courbure CAD
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 24)

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)


    #gmsh.model.geo.synchronize() -> To be deleted
    gmsh.write(model_name + ".msh")

    mesh_data = gmshio.model_to_mesh(
        model, comm, model_rank
    )
    final_mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = mesh_data.cell_tags
    #gmsh.fltk.run()
    gmsh.finalize()

    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def curved_cubic_domain_CAD(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):

    gmsh.initialize()
    
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("from_comsol")
    if side_box == 0.11:
        case = "small"
    elif side_box == 0.40:
        case = "large"
    else:
        raise ValueError("side_box must be either 0.11 or 0.40")

    gmsh.model.occ.importShapes(f"{case}_object_curved_cubic_case.step")
    gmsh.model.occ.synchronize()
    
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)

    
    volumes = gmsh.model.getEntities(dim=3)
    print(f'volumes: {volumes}')
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [5, 6, 7, 8, 9, 10, 11], tag=3)


    
    # Adaptation à la courbure CAD
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 24)

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)


    #gmsh.model.geo.synchronize() -> To be deleted
    gmsh.write(model_name + ".msh")

    mesh_data = gmshio.model_to_mesh(
        model, comm, model_rank
    )
    final_mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = mesh_data.cell_tags
    #gmsh.fltk.run()
    gmsh.finalize()

    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def spherical_domain_CAD(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):

    gmsh.initialize()
    
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("from_comsol")
    if side_box == 0.11:
        case = "small"
    elif side_box == 0.40:
        case = "large"
    else:
        raise ValueError("side_box must be either 0.11 or 0.40")

    gmsh.model.occ.importShapes(f"{case}_object_spherical_case.step")
    gmsh.model.occ.synchronize()
    
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)

    
    volumes = gmsh.model.getEntities(dim=3)
    print(f'volumes: {volumes}')
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [3], tag=3)


    
    # Adaptation à la courbure CAD
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 6)

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)


    #gmsh.model.geo.synchronize() -> To be deleted
    gmsh.write(model_name + ".msh")

    mesh_data = gmshio.model_to_mesh(
        model, comm, model_rank
    )
    final_mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = mesh_data.cell_tags
    #gmsh.fltk.run()
    gmsh.finalize()

    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def new_broken_cubic_domain_CAD_side_box_sweep(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):

    gmsh.initialize()
    
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("from_comsol")

    side_box_str = f"{side_box:.2f}".replace(".", "p")
    gmsh.model.occ.importShapes(f"./export_geoCOMSOL/geometry_side_box_{side_box_str}_mm.step")
    gmsh.model.occ.synchronize()
    
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)

    
    volumes = gmsh.model.getEntities(dim=3)
    print(f'volumes: {volumes}')
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15], tag=3)


    
    # Adaptation à la courbure CAD
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)


    #gmsh.model.geo.synchronize() -> To be deleted
    gmsh.write(model_name + ".msh")

    mesh_data = gmshio.model_to_mesh(
        model, comm, model_rank
    )
    final_mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = mesh_data.cell_tags
    #gmsh.fltk.run()
    gmsh.finalize()

    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info


def biSpherical_domain_CAD(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):

    gmsh.initialize()
    
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("from_comsol")

    gmsh.model.occ.importShapes("biSphericalComplex.step")
    gmsh.model.occ.synchronize()
    
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)

    
    volumes = gmsh.model.getEntities(dim=3)
    print(f'volumes: {volumes}')
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    #gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [1, 2, 3, 4, 5, 6, 7, 8], tag=3)


    
    # Adaptation à la courbure CAD
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)


    #gmsh.model.geo.synchronize() -> To be deleted
    gmsh.write(model_name + ".msh")

    mesh_data = gmshio.model_to_mesh(
        model, comm, model_rank
    )
    final_mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = mesh_data.cell_tags
    #gmsh.fltk.run()
    gmsh.finalize()

    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [-0.05, -0.05, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info


def cubicSpherical_domain_CAD(side_box = 0.2, radius = 0.02, lc = 8e-3, model_name = "broken_cubic"):

    gmsh.initialize()
    
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("from_comsol")

    gmsh.model.occ.importShapes("cubicSphericalComplex.step")
    gmsh.model.occ.synchronize()
    
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)

    
    volumes = gmsh.model.getEntities(dim=3)
    print(f'volumes: {volumes}')
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    #gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [i+1 for i in range(26)], tag=3)


    
    # Adaptation à la courbure CAD
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 24)

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)


    #gmsh.model.geo.synchronize() -> To be deleted
    gmsh.write(model_name + ".msh")

    mesh_data = gmshio.model_to_mesh(
        model, comm, model_rank
    )
    final_mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = mesh_data.cell_tags
    #gmsh.fltk.run()
    gmsh.finalize()

    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [-0.05, -0.05, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    entity_maps_mesh = [entity_map]

    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info