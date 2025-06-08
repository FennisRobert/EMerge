import fem
import gmsh
from scipy.sparse.linalg import gmres
from scipy.spatial import Delaunay

import pyescher as pe
import numpy as np

from fem.elements.legrange2 import Legrange2

mm = 0.001
wga = 22.86*mm
L = 70*mm
wgb = 10.16*mm
np.set_printoptions(precision=3, suppress=True)
viewer = pe.Viewer()

Nmodes = 1
with fem.Simulation3D('MySimulation') as model:
    model.physics.set_order(2)
    box1 = fem.geo3d.Box(L, wga, wgb)
    box2 = fem.geo3d.Box(wga, L, wgb, position=(L-wga, 0,0))
    box1 = fem.geo3d.add(box1, box2)

    model.physics.resolution = 0.2
    
    model.physics.set_frequency(10e9)

    model.define_geometry(box1)

    model.generate_mesh()
    
    port1 = fem.bc.RectangularWaveguide(model.select.face.near(0, wga/2, wgb/2), 1, True)
    port2 = fem.bc.RectangularWaveguide(model.select.face.near(L-wga/2, L-wga/2, wgb/2), 2, False)

    model.physics.assign(port1)
    model.physics.assign(port2)

    model.physics.set_frequency(np.linspace(8.5e9,11.5e9,2))

    data = model.run_frequency_domain()

    xs = np.linspace(0, L, 41)
    ys = np.linspace(0, L, 41)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs = X.flatten()
    ys = Y.flatten()
    zs = np.ones_like(xs) * wgb*0.45

    dataset = fem.Dataset2D(data[0], np.array([xs,ys]), Delaunay(np.array([xs, ys]).T).simplices.T)
    
    S11 = data.glob('S11')[:]
    S21 = data.glob('S21')[:]

    print('S11:', 20*np.log10(np.abs(S11)))
    print('S21:', 20*np.log10(np.abs(S21)))
    print('Sum:', np.abs(S11)**2 + np.abs(S21)**2)
    
    for i in range(2):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = model.physics.basis.interpolate(data[i], xs, ys, zs)
        
        dataset.Ex = Ex
        dataset.Ey = Ey
        dataset.Ez = Ez

        Ex = np.real(Ex)
        Ey = np.real(Ey)
        Ez = np.real(Ez)

        fem.plot.animate_field(dataset, dataset.Ez, 35)
        ex, ey, ez = model.mesh.edge_centers

        nE = model.mesh.n_edges
        nT = model.mesh.n_tris

        ds = 0.0001
        with viewer.new3d('Solution') as v:
            
            v.quiver3d(xs, ys, zs, Ex, Ey, Ez)
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(port1.tags)])
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(port2.tags)])

        input('Press Enter to continue...')

