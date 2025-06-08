import fem
from scipy.spatial import Delaunay
import pyescher as pe
import numpy as np


mm = 0.001
W = 10*mm
th = 0.5*mm
H = 8*mm
L = 30*mm

wline = 2*mm
er = 2

margin = 5*mm

np.set_printoptions(precision=3, suppress=True)
viewer = pe.Viewer()

Nmodes = 1

f1 = 5e9
f2 = 6e9
with fem.Simulation3D('MySimulation') as model:
    model.physics.set_order(2)
    dielectric = fem.geo3d.Box(W, L, th, position=(-W/2, 0, -th))
    air = fem.geo3d.Box(W, L, H, position=(-W/2, 0, 0))
    centre = fem.geo3d.XYPlate(wline, L-2*margin, position=(-wline/2, margin, 0))
    
    surfport1 = fem.geo3d.Plate(np.array([-wline/2,margin,-th]), np.array([wline,0,0]), np.array([0,0,th]))
    surfport2 = fem.geo3d.Plate(np.array([-wline/2,L-margin,-th]), np.array([wline,0,0]), np.array([0,0,th]))

    model.physics.resolution = 0.15
    
    model.physics.set_frequency(f2)

    model.define_geometry([air, dielectric, centre, surfport1, surfport2])

    dielectric.material = fem.material.Material(er) 

    model.mesher.set_boundary_size(centre.dimtags, 1*mm, edge_only=True)

    model.mesher.set_boundary_size(surfport1.dimtags, 1*mm)
    model.mesher.set_boundary_size(surfport2.dimtags, 1*mm)


    model.generate_mesh()
    model.mesh.plot_gmsh()
    
    port1 = fem.bc.LumpedPort(model.select.obj(surfport1), 1, width=wline, height=th, direction=fem.ZAX, active=True, Z0=50)
    port2 = fem.bc.LumpedPort(model.select.obj(surfport2), 2, width=wline, height=th, direction=fem.ZAX, active=False, Z0=50)

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
    
    model.physics.set_frequency(np.linspace(f1, f2,2))

    data = model.run_frequency_domain()

    xs = np.linspace(-W/2, W/2, 11)
    ys = np.linspace(0, L, 21)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = -np.ones_like(X) * th/2
    #dataset = fem.Dataset2D(data[0], np.array([xs,ys]), Delaunay(np.array([xs, ys]).T).simplices.T)
    
    S11 = data.glob('S11')[:]
    S21 = data.glob('S21')[:]

    print('S11:', 20*np.log10(np.abs(S11)))
    print('S21:', 20*np.log10(np.abs(S21)))
    print('Sum:', np.abs(S11)**2 + np.abs(S21)**2)

    print(f'Phase Shift = {np.angle(S21)*180/np.pi} deg')
    freqs = np.array(model.physics.frequencies)
    print(f'Expected phase shift = {180/np.pi*np.angle(np.exp(1j*(-2*np.pi*freqs/299792458)*L*np.sqrt(3.079)))}')

    for i in range(2):
        nodes = model.mesh.nodes
        
        Ex, Ey, Ez = data.item(i).interpolate(X,Y,Z).E
        
        xp1, yp1, zp1 = fem.XZPLANE.grid(np.linspace(-wline/2, wline/2, 11), np.linspace(-th, 0, 5), (0,margin,0))
        Ep1x, Ep1y, Ep1z = data.item(i).interpolate(xp1, yp1, zp1).E

        fem.plot.animate_field(X,Y, Ez, 35)
        ex, ey, ez = model.mesh.edge_centers

        nE = model.mesh.n_edges
        nT = model.mesh.n_tris

        ds = 0.0001
        with viewer.new3d('Solution') as v:
            
            #v.quiver3d(xs, ys, zs, Ex, Ey, Ez)
            v.quiver3d(xp1, yp1, zp1, Ep1x.real, Ep1y.real, Ep1z.real)
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(port1.tags)])
            v.mesh(nodes, model.mesh.tris[:,model.mesh.get_triangles(port2.tags)])

        input('Press Enter to continue...')

