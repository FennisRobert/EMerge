import fem
import pyvista as pv
from fem.plotting.pyvista import Display
import numpy as np
np.set_printoptions(precision=3, linewidth=150)
mm = 0.001
A = 22.86*mm
L = 20*mm
H = 10.16*mm
th = 1*mm

Nmodes = 50

with fem.Simulation3D('BMATest', 'DEBUG') as m:
 
    wg = fem.modeling.Box(A,L,H,(0,0,0))
    
    m.physics.resolution = 0.2
    m.physics.set_frequency(9e9)

    m.define_geometry([wg,])
    m.generate_mesh()
    port1 = fem.bc.ModalPort(m.select.face.near(A/2, L, H/2), 1, True)
    m.physics.assign(port1)

    
    data = m.physics.modal_analysis(port1, Nmodes, TEM=False,
                                    direct=True)
    
    x = np.linspace(0,A, 41)
    y = np.linspace(0,H, 31)
    X,Z = np.meshgrid(x,y)
    Y = 0*X+L
    for i in range(port1.nmodes):
        port1.selected_mode = i
        mode = port1.get_mode(i)
        p = pv.Plotter()
        d = Display(m.mesh, p)
        d.add_mesh(fem.FaceSelection(m.physics.boundary_conditions[0].tags), opacity=1, show_edges=True, edge_opacity=1)
        d.plot_portmode(port1, data.item(i).k0, 61, dv=(0,0.1*mm,0), XYZ = (X,Y,Z))
        p.show()

        p = pv.Plotter()
        d = Display(m.mesh, p)
        d.add_mesh(fem.FaceSelection(m.physics.boundary_conditions[0].tags), opacity=1, show_edges=True, edge_opacity=1)
        d.plot_portmode(port1, data.item(i).k0, 61, dv=(0,0.1*mm,0), XYZ = (X,Y,Z), field='H')
        p.show()
