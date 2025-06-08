from emerge.solvers import fem
import gmsh
from scipy.sparse.linalg import gmres
from scipy.spatial import Delaunay

import pyescher as pe
import numpy as np

mm = 0.001
wga = 22.86*mm
L = 50*mm
wgb = 10.16*mm
np.set_printoptions(precision=3, suppress=True)
viewer = pe.Viewer()

with fem.Simulation3D('MySimulation') as model:
    box1 = fem.geo3d.Box(L, wga, wgb)
    box2 = fem.geo3d.Box(2*mm, wga, wgb/2, position=(L/2-mm,0,0))

    box1 = fem.geo3d.subtract(box1, box2)

    
    model.physics.set_frequency(10e9)

    model.resolution = 0.1

    model.define_geometry(box1)

    
    p1 = fem.bc.Port(port_number=1, active=True, vectors=fem.YZPLANE)
    p2 = fem.bc.Port(port_number=2, active=False, vectors=fem.YZPLANE)

    model.physics.assign(p1, domain=model.geo.select(0, wga/2,wgb/2).face)
    model.physics.assign(p2, domain=model.geo.select(L, wga/2,wgb/2).face)

    model.generate_mesh()
    
    model.mesh.plot_gmsh()

    model.physics.set_frequency(np.linspace(8.5e9,11.5e9,2))

    data = model.run_frequency_domain(solver=gmres)

    xs = np.linspace(0, L, 41)
    ys = np.linspace(0, wga, 21)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs = X.flatten()
    ys = Y.flatten()
    zs = np.ones_like(xs) * wgb*0.45

    dataset = fem.Dataset2D(data.freqs, np.array([xs,ys]), Delaunay(np.array([xs, ys]).T).simplices.T)
    

    for i in range(2):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = data(i).interpolate(xs, ys, zs)
        

        dataset.Ex = Ex
        dataset.Ey = Ey
        dataset.Ez = Ez

        Ex = np.abs(Ex)
        Ey = np.abs(Ey)
        Ez = np.abs(Ez)

        fem.plot.animate_field(dataset, dataset.Ez, 35)
        ex, ey, ez = model.mesh.edge_centers
        with viewer.new3d('Solution') as v:
            v.scatter(nodes[0,:], nodes[1,:], nodes[2,:], size=0.0001)
            v.quiver3d(xs, ys, zs, Ex, Ey, Ez)
            v.mesh(nodes, model.mesh.tris[:,p1.indices])
            v.mesh(nodes, model.mesh.tris[:,p2.indices])
            v.quiver3d(ex[p1._edge_ids], ey[p1._edge_ids], ez[p1._edge_ids], p1._b[0,:], p1._b[1,:], p1._b[2,:], color=(0,0,1))

        input('Press Enter to continue...')
    #gmsh.fltk.run()

