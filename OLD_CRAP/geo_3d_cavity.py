from emerge.solvers import fem
import gmsh
import pyescher as pe
import numpy as np
mm = 0.001
wga = 22.86*mm
L = 30*mm
wgb = 10.16*mm
np.set_printoptions(precision=3, suppress=True)
viewer = pe.Viewer()
with fem.Simulation3D('MySimulation') as model:
    box1 = fem.geo3d.Box(L, L, wgb)
    #box2 = fem.geo3d.Box(L, wga, wgb, position=(L,0,0))
    model.physics.set_frequency(10e9)

    model.resolution = 0.1
    model.define_geometry([box1,])

    model.generate_mesh()

    data = model.run_eigenmode(num_sols=8)

    bnodes = model.physics.boundary_conditions[0]
    face_tags = bnodes.tags
    tri_ids = model.mesh.get_triangles(face_tags)
    edge_ids = list(model.mesh.tri_to_edge[:,tri_ids].flatten())
    ex, ey, ez = model.mesh.edge_centers[0,:], model.mesh.edge_centers[1,:], model.mesh.edge_centers[2,:]
    
    xs = np.linspace(0, L, 21)
    ys = np.linspace(0, L, 21)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs = X.flatten()
    ys = Y.flatten()
    zs = np.ones_like(xs) * wgb/2

    
    for i in range(4):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = data(i).interpolate(xs, ys, zs)
        
        
        with viewer.new3d('Solution') as v:
            v.scatter(nodes[0,:], nodes[1,:], nodes[2,:], size=0.0001)
            v.quiver3d(xs, ys, zs, Ex, Ey, Ez)
            #v.scatter(ex[edge_ids], ey[edge_ids], ez[edge_ids], size=0.001)
        input('Press Enter to continue...')
    #gmsh.fltk.run()

