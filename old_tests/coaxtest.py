import fem
import numpy as np
import pyescher as pe

import pyvista as pv

from fem.plotting.pyvista import Display
mm = 0.001

rout = 5*mm
rin = 3*mm
L = 15*mm

f0 = 20e9

#view = pe.Viewer()

with fem.Simulation3D('CoaxCable') as m:
    
    coax_cs = fem.CoordinateSystem(
        fem.XAX, fem.YAX, fem.ZAX, origin=(0, 0, 0)
    )
    cyl = fem.modeling.CoaxCyllinder(
        rout=rout,
        rin=rin,
        height=L,
        cs=coax_cs
    )
    
    cyl.max_meshsize = 2*mm

    m.physics.set_frequency(f0)

    m.define_geometry([cyl])

    m.generate_mesh()

    #m.mesh.plot_gmsh()

    portface = m.select.face.near(0,0,0)

    port1 = fem.bc.ModalPort(
        m.select.face.near(0,0,0),
        1, 
        True,
        coax_cs,
        vintline=fem.Line(np.array([rin, rout]), 
                          np.array([0, 0]), 
                          np.array([0, 1])),

    )

    m.physics.assign(port1)

    data = m.physics.modal_analysis(port1, 10, direct=True, TEM =True)

    #xs, ys, zs = cyl.face_points(nRadius=2, Angle=20, face_number=1)

    #Ex, Ey, Ez = data(mode=1).interpolate(xs, ys, zs, freq=f0).E
    
    for i in range(10):
        p = pv.Plotter()
        port1.selected_mode = i
        d = Display(m.mesh, p)
        d.plot_portmode(port1, data.item(i).k0, Npoints=51)
        p.show()