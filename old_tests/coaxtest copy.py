import fem
import numpy as np
import pyescher as pe
import pyvista as pv

from fem.plotting.pyvista import Display
mm = 0.001

rout = 10*mm
L = 15*mm

f0 = 10e9

w = 2*mm
out = 3*mm


with fem.Simulation3D('CoaxCable') as m:
    
    coax_cs = fem.CoordinateSystem(
        fem.XAX, fem.ZAX, fem.YAX, origin=(0, 0, 0)
    )
    cyl = fem.modeling.Cyllinder(
        radius=rout,
        height=L,
        cs=coax_cs
    )
    cutblock1 = fem.modeling.Box(w, L, rout-out, position=(-w/2, 0, out))
    cutblock2 = fem.modeling.Box(w, L, rout-out, position=(-w/2, 0, out))
    cutblock3 = fem.modeling.Box(w, L, rout-out, position=(-w/2, 0, out))
    
    fem.modeling.rotate(cutblock2, (0,0,0), (0,1,0), 120)
    fem.modeling.rotate(cutblock3, (0,0,0), (0,1,0), -120)

    cyl = fem.modeling.subtract(cyl, cutblock1)
    cyl = fem.modeling.subtract(cyl, cutblock2)
    cyl = fem.modeling.subtract(cyl, cutblock3)

    cyl.max_meshsize = 2*mm

    m.physics.set_frequency(f0)

    m.define_geometry([cyl])

    m.generate_mesh()

    m.mesh.plot_gmsh()

    portface = m.select.face.near(0,0,0)

    port1 = fem.bc.ModalPort(
        m.select.face.near(0,0,0),
        1, 
    )

    m.physics.assign(port1)

    data = m.physics.modal_analysis(port1, 4)

    #xs, ys, zs = cyl.face_points(nRadius=2, Angle=20, face_number=1)

    #Ex, Ey, Ez = data(mode=1).interpolate(xs, ys, zs, freq=f0).E
    
    for i in range(4):
        p = pv.Plotter()
        port1.selected_mode = i
        d = Display(m.mesh, p)
        d.plot_portmode(port1, data.item(i).k0, Npoints=20)
        p.show()