import fem
from scipy.spatial import Delaunay
import pyescher as pe
import numpy as np


mm = 0.001
W = 10*mm
th = 1*mm
H = 8*mm
L = 30*mm

wline = 2*mm

np.set_printoptions(precision=3, suppress=True)
viewer = pe.Viewer()

Nmodes = 1
f1 = 5e9
f2 = 6e9
with fem.Simulation3D('MySimulation') as model:
    model.physics.set_order(2)
    dielectric = fem.geo3d.Box(W, L, th, position=(-W/2, 0, -th))
    air = fem.geo3d.Box(W, L, H, position=(-W/2, 0, 0))
    centre = fem.geo3d.XYPlate(wline, L, position=(-wline/2, 0, 0))
    
    surfport1 = fem.geo3d.Plate(np.array([-W*0.8/2,0,-th]), np.array([W*0.8,0,0]), np.array([0,0,0.5*H]))
    surfport2 = fem.geo3d.Plate(np.array([-W*0.8/2,L,-th]), np.array([W*0.8,0,0]), np.array([0,0,0.5*H]))

    model.physics.resolution = 0.3
    
    model.physics.set_frequency(f2)

    model.define_geometry([air, dielectric, centre, surfport1, surfport2])

    dielectric.material = fem.material.Material(4)
    # model.mesher.set_material(dielectric, fem.material.Material(4))

    model.mesher.set_boundary_size(centre.dimtags, 1*mm, edge_only=True)

    model.mesher.set_boundary_size(surfport1.dimtags, 1*mm)
    model.mesher.set_boundary_size(surfport2.dimtags, 1*mm)


    model.generate_mesh()
    
    #model.mesh.plot_gmsh()
    intline1 = fem.Line.from_points(np.array([0, 0, -th]), np.array([0, 0, 0]), 11)
    intline2 = fem.Line.from_points(np.array([0, L, -th]), np.array([0, L, 0]), 11)

    port1 = fem.bc.ModalPort(model.select.obj(surfport1), 1, True, vintline=intline1)
    port2 = fem.bc.ModalPort(model.select.obj(surfport2), 2, False, vintline=intline2)
    pec = fem.bc.PEC(model.select.obj(centre))

    model.physics.assign(port1)
    model.physics.assign(port2)
    model.physics.assign(pec)

    with viewer.new3d('BC View') as v:
        for bc in [port1, port2, pec]:

            tri_ids = model.mesh.get_triangles(bc.tags)
            
            nodes = model.mesh.nodes
            tris = model.mesh.tris[:, tri_ids]
            v.mesh(nodes, tris)
    
    model.physics.modal_analysis(port1, 1, True)
    model.physics.modal_analysis(port2, 1, True)

    model.physics.set_frequency(np.linspace(f1,f2,2))

    data = model.run_frequency_domain()

    xs = np.linspace(-W/2, W/2, 11)
    ys = np.linspace(0, L, 21)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs = X.flatten()
    ys = Y.flatten()
    zs = -np.ones_like(xs) * th/2

    dataset = fem.Dataset2D(data[0], np.array([xs,ys]), Delaunay(np.array([xs, ys]).T).simplices.T)
    
    S11 = data.glob('S11')[:]
    S21 = data.glob('S21')[:]

    print('S11:', 20*np.log10(np.abs(S11)))
    print('S21:', 20*np.log10(np.abs(S21)))
    print('Sum:', np.abs(S11)**2 + np.abs(S21)**2)
    


    for i in range(2):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = data.item(i).interpolate(xs, ys, zs).E
        
        xp1, yp1, zp1 = fem.XZPLANE.grid(np.linspace(-W/2, W/2, 21), np.linspace(-th, 2*th, 11), (0,0,0))
        Ep1x, Ep1y, Ep1z = data.item(i).interpolate(xp1, yp1, zp1).E

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
            v.quiver3d(xp1, yp1, zp1, Ep1x.real, Ep1y.real, Ep1z.real)
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(port1.tags)])
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(port2.tags)])

        input('Press Enter to continue...')

