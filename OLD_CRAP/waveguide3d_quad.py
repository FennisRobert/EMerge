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
    box2 = fem.geo3d.Box(L, 0.3*wga, 0.3*wgb, position=(0,0.35*wga,0.35*wgb))
    model.physics.resolution = 0.2
    box1 = fem.geo3d.subtract(box1, box2)
    
    model.physics.set_frequency(10e9)
    #model.physics.solveroutine.solver = fem.SolverAMG()
    model.define_geometry(box1)

    p1 = fem.bc.ModalPort(port_number=1, active=True, basis=fem.YZPLANE, dims=(wga, wgb), origin=(0, 0, 0))
    p2 = fem.bc.ModalPort(port_number=2, active=False, basis=fem.YZPLANE, dims=(wga, wgb), origin=(L, 0, 0))
    
    model.physics.assign(p1, domain=model.mesher.select(0, wga/2,wgb/2).face)
    model.physics.assign(p2, domain=model.mesher.select(L, wga/2,wgb/2).face)

    model.generate_mesh()
    
    #model.mesh.plot_gmsh()

    model.physics.set_frequency(np.linspace(1e9,11.5e9,2))

    data1 = model.physics.modal_analysis(p1, nmodes=1, TEM=True)
    data2 = model.physics.modal_analysis(p2, nmodes=1, TEM=True)

    model.physics.set_frequency(np.linspace(8.5e9,11.5e9,2))

    data = model.run_frequency_domain()
    
    ys = np.linspace(0, wga, 21)
    zs = np.linspace(0, wgb, 21)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    Y = Y.flatten()
    Z = Z.flatten()
    X = 0* np.ones_like(Y)
    
    pec_ids = model.physics.basis._pec_ids
    port_ids = model.physics.basis._solve_ids

    nE = model.physics.basis.n_edges
    nT = model.physics.basis.n_tris

    pec_edge_ids = pec_ids[pec_ids < nE]
    pec_tri_ids = pec_ids[(pec_ids >= nE) & (pec_ids < nE+nT)] - nE
    port_edge_ids = port_ids[port_ids < nE]
    port_tri_ids = port_ids[(port_ids >= nE) & (port_ids < nE+nT)] - nE
    
    nx, ny, nz = model.physics.mesh.nodes[0,:], model.physics.mesh.nodes[1,:], model.physics.mesh.nodes[2,:]
    ex, ey, ez = model.physics.mesh.edge_centers[0,:], model.physics.mesh.edge_centers[1,:], model.physics.mesh.edge_centers[2,:]
    tx, ty, tz = model.physics.mesh.tri_centers[0,:], model.physics.mesh.tri_centers[1,:], model.physics.mesh.tri_centers[2,:]
    

    xs = np.linspace(0, L, 41)
    ys = np.linspace(0, wga, 21)
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
    
    pys = np.linspace(0, wga, 11)
    pzs = np.linspace(0, wgb, 11)
    PY, PZ = np.meshgrid(pys, pzs, indexing='ij')
    pys = PY.flatten()
    pzs = PZ.flatten()
    pxs = np.ones_like(pys) * L


    for i in range(2):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = model.physics.basis.interpolate(data[i], xs, ys, zs)
        
        dataset.Ex = Ex
        dataset.Ey = Ey
        dataset.Ez = Ez

        Ex = np.real(Ex)
        Ey = np.real(Ey)
        Ez = np.real(Ez)

        PEx, PEy, PEz = model.physics.basis.interpolate(data[i], pxs, pys, pzs)

        PEx = np.real(PEx)
        PEy = np.real(PEy)
        PEz = np.real(PEz)

        fem.plot.animate_field(dataset, dataset.Ey, 35)
        ex, ey, ez = model.mesh.edge_centers

        b = p1._b
        
        nE = model.mesh.n_edges
        nT = model.mesh.n_tris

        ds = 0.0001
        with viewer.new3d('Solution') as v:
            
            v.quiver3d(xs, ys, zs, Ex, Ey, Ez)
            v.quiver3d(pxs, pys, pzs, PEx, PEy, PEz)
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(p1.tags)])
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(p2.tags)])

        input('Press Enter to continue...')