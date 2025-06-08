import fem
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
    model.physics.set_order(2)
    model.physics.set_frequency(9e9)

    model.resolution = 0.3

    model.define_geometry([box1,])

    model.generate_mesh()

    data = model.run_eigenmode(num_sols=8)

    bnodes = model.physics.boundary_conditions[0]
    face_tags = bnodes.tags
    tri_ids = model.mesh.get_triangles(face_tags)
    edge_ids = list(model.mesh.tri_to_edge[:,tri_ids].flatten())
    ex, ey, ez = model.mesh.edge_centers[0,:], model.mesh.edge_centers[1,:], model.mesh.edge_centers[2,:]
    
    xs = np.linspace(0, L, 41)
    ys = np.linspace(0, L, 41)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs = X.flatten()
    ys = Y.flatten()
    zs = np.ones_like(xs) * wgb/2

    
    for i in range(8):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = data(i).interpolate(xs, ys, zs)
        nE = model.mesh.n_edges
        nT = model.mesh.n_tris

        Esoledge = data(i)[:nE]
        Esoltri = data(i)[nE:nE+nT]
        Esoledge2 = data(i)[nE+nT:nE+nT+nE]
        Esoltri2 = data(i)[nE+nT+nE:]

        Esoledge = Esoledge + Esoledge2
        Esoltri = Esoltri + Esoltri2
        ex = model.mesh.edge_centers[0,:]
        ey = model.mesh.edge_centers[1,:]
        ez = model.mesh.edge_centers[2,:]

        tx = model.mesh.tri_centers[0,:]
        ty = model.mesh.tri_centers[1,:]
        tz = model.mesh.tri_centers[2,:]
        with viewer.new3d('Solution') as v:
            v.scatter(nodes[0,:], nodes[1,:], nodes[2,:], size=0.0001)
            v.quiver3d(xs, ys, zs, Ex, Ey, Ez)
            #v.scatter(ex, ey, ez, s=np.abs(Esoledge))
            #v.scatter(tx, ty, tz, np.abs(Esoltri))
        input('Press Enter to continue...')
    #gmsh.fltk.run()

