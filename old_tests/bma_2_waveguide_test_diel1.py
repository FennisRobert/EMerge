import fem
import pyvista as pv
from fem.plotting.pyvista import Display
import numpy as np

mm = 0.001
A = 22.86*mm
L = 20*mm
H = 10.16*mm
th = 1*mm

Nmodes = 30

with fem.Simulation3D('BMATest') as m:

    #coax = fem.modeling.CoaxCyllinder(2*mm, 1*mm, 3*mm, fem.cs.GCS)  
      
    wgL = fem.modeling.Box(A/2,L,H,(0,0,0))
    wgR = fem.modeling.Box(A/2,L,H,(A/2,0,0))
    wgL.material = fem.material.Material(4)

    port = fem.modeling.Plate(np.array([0,L,0]),
                              np.array([A,0,0]),
                              np.array([0,0,H]))
    #inner = fem.modeling.Box(A/3,L,H/3,(A/3,0,H/3))

    #strip = fem.modeling.XYPlate(3*mm, L, (-1.5*mm,0,0))

    #wg = fem.modeling.subtract(wg,inner)
    # port = fem.modeling.Plate(np.array([0,0,0]),
    #                            np.array([A,0,0]),
    #                            np.array([0,0,H]))
    
    m.physics.resolution = 0.15
    m.physics.set_frequency(6e9)

    m.define_geometry([wgL,wgR, port])
    m.generate_mesh()
    port1 = fem.bc.ModalPort(port.select, 1, True)
    m.physics.assign(port1)

    #m.physics.assign(port1,pec)

    data = m.physics.modal_analysis(port1, Nmodes, TEM=False,
                                    direct=True)
    
    x = np.linspace(0,A, 41)
    y = np.linspace(0,H, 31)
    X,Z = np.meshgrid(x,y)
    Y = 0*X+L
    for i in range(port1.nmodes):
        port1.selected_mode = i
        print(port1.get_mode(i))
        p = pv.Plotter()
        d = Display(m.mesh, p)
        d.add_mesh(fem.FaceSelection(m.physics.boundary_conditions[0].tags), opacity=1, show_edges=True, edge_opacity=1)
       # d.add_mesh(strip)
        d.plot_portmode(port1, data.item(i).k0, 61, dv=(0,0.1*mm,0), XYZ = (X,Y,Z))
        p.show()
