from __future__ import annotations

import fem
import pyvista as pv
from fem.plotting.pyvista import Display
import numpy as np

mm = 0.001
A = 22.86*mm
L = 20*mm
H = 10.16*mm
th = 2*mm

Nmodes = 5

wstrip = 4*mm

class Test:

    def __init__(self):
        self.a = 3

    def increase(self, amount) -> Test:
        self.a += amount
        return self
    
with fem.Simulation3D('BMATest') as m:
     
    diel = fem.modeling.Box(A,L,th,(-A/2,0,-th))
    diel.material = fem.material.Material(4)
    diel.max_meshsize = 1*mm
    air = fem.modeling.Box(A,L,H-th,(-A/2,0,0))

    #strip = fem.modeling.Box(wstrip, L, 5*mm, (-wstrip/2, 0, 0))

    #air = fem.modeling.subtract(air, diel, remove_tool=False)
    strip = fem.modeling.XYPlate(wstrip, L, (-wstrip/2,0,0))

   
    port = fem.modeling.Plate(np.array([-A/2,L,-th]),
                               np.array([A,0,0]),
                               np.array([0,0,H]))
    
    
    m.physics.resolution = 0.1
    m.physics.set_frequency(5e9)

    m.define_geometry([diel,air,port, strip])
    m.mesher.set_boundary_size(strip.dimtags, 3*mm, edge_only=False)
    #m.mesher.set_boundary_size(m.select.face.near(0,0,0), 2*mm)
    m.generate_mesh()
    #m.mesh.plot_gmsh()
    #port1 = fem.bc.ModalPort(port.select, 1, True)
    port1 = fem.bc.ModalPort(m.select.face.inlayer(0,L-1*mm, 0, np.array([0,5*mm, 0])), 1, True)
    pec = fem.bc.PEC(strip)
    m.physics.assign(port1, pec)

    #m.physics.assign(port1,pec)

    data = m.physics.modal_analysis(port1, Nmodes, TEM=True, direct=True)
    
    x = np.linspace(-A/2,A/2, 41)
    y = np.linspace(-th,H-th, 31)
    X,Z = np.meshgrid(x,y)
    Y = 0*X+L

    pec_edges = np.array([port1._pece]).squeeze()
    pec_vertices = np.array([port1._pecv]).squeeze()
    pexyz = port1._field.mesh.edge_centers[:,pec_edges]
    pvxyz = port1._field.mesh.nodes[:,pec_vertices]
    for i in range(Nmodes):
        port1.selected_mode = i
        print(port1.get_mode(i))
        p = pv.Plotter()
        d = Display(m.mesh, p)
        d.add_mesh(fem.FaceSelection(m.physics.boundary_conditions[0].tags), opacity=1, show_edges=True, edge_opacity=1)
        #d.add_mesh(port, opacity=)
        d.add_scatter(pexyz[0,:], pexyz[1,:], pexyz[2,:])
        d.add_scatter(pvxyz[0,:], pvxyz[1,:], pvxyz[2,:])
        d.plot_portmode(port1, data.item(i).k0, 61, dv=(0,-0.1*mm,0), XYZ = (X,Y,Z))
        p.show()
