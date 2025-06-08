import fem
import numpy as np
from fem.plotting.pyvista import Display
import pyvista as pv
import pyescher as pe

mm = 0.001
W = 10*mm
th = 1.2*mm
H = 8*mm
L = 30*mm

wline = 2*mm

w1 = w7 = 5.0*mm
w2 = w6 = 0.5*mm
w3 = w5 = 5.0*mm
w4 = 0.5*mm

l1 = l7 = 3.3*mm
l2 = l6 = 8.4*mm
l3 = l5 = 13.4*mm
l4 = 13.5*mm

w0, l0 = 2.3*mm, 5.0*mm


np.set_printoptions(precision=3, suppress=True)

margin = 5*mm
Nmodes = 1
f1 = 0.5e9
f2 = 3e9
W = 4*max(w0, w1, w2, w3, w4, w5, w6, w7)
L = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l0 + 2*margin

with fem.Simulation3D('MySimulation') as model:
    dielectric = fem.modeling.Box(W, L, th, position=(-W/2, -margin, -th))
    air = fem.modeling.Box(W, L, H, position=(-W/2, -margin, 0))
    
    pcb = fem.modeling.PCBLayouter(W, L, th, 0.02, origin=(0, 0, 0), unit=1)

    pcb.new(-l0, 0, w0, (1,0)).straight(l0,w0).turn(-90, w0).straight(l0/3, w0).straight(l1, w1).straight(l2, w2)\
        .straight(l3, w3).straight(l4, w4).straight(l5, w5).straight(l6, w6).straight(l7, w7)\
            .straight(l0/4, w0).turn(-90,w0).straight(l0,w0)

    pcb.lumped_port(pcb(0).start)
    pcb.lumped_port(pcb(0).end)

    pcb.compile_paths(z=0)
    
    dielectric.material = fem.material.Material(4.4)
    model.physics.resolution = 0.3
    
    model.physics.set_frequency(5e9)

    model.define_geometry([dielectric, air] + pcb.all_objects)
    
    model.mesher.set_boundary_size(pcb.traces[0].dimtags, 2*mm, edge_only=True)

    model.generate_mesh()
    
    port1 = fem.bc.LumpedPort(pcb.ports[0], 1, width=w0, height=th, direction=fem.ZAX, active=True, Z0=50)
    port2 = fem.bc.LumpedPort(pcb.ports[1], 2, width=w0, height=th, direction=fem.ZAX, active=False, Z0=50)

    pec = fem.bc.PEC(pcb.trace)

    model.physics.assign(port1, port2, pec)

    model.physics.set_frequency(np.linspace(f1, f2, 21))

    data = model.physics.frequency_domain()

    X, Y, Z = fem.XYPLANE.grid((-W/2,W/2,21), (-margin, L-margin, 41), (0,0,-th/2))
    
    Ex, Ey, Ez = data.item(1).interpolate(X,Y,Z).E

    xyz = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    E = np.array([Ex.flatten(), Ey.flatten(), Ez.flatten()])
    model.step4()
    
    freq, S21 = data.ax('freq').S(2,1)
    freq, S11 = data.ax('freq').S(1,1)

    pe.plot_lines(pe.Line(freq*1e-9, S11, label='S11'), pe.Line(freq*1e-9, S21, label='S21'),
                  transformation = pe.dB)
    print(data.ax('freq').S(2,1))
    p = pv.Plotter()
    disp = Display(model.mesh, p)

    disp.add_mesh(dielectric, color='green', show_edges=True, opacity=0.1)
    disp.add_mesh(air, color='lightblue', show_edges=True, opacity=0.1)
    disp.add_mesh(pcb.all_objects, color='red')

    p.add_mesh(pv.StructuredGrid(X,Y,Z), scalars=np.abs(Ez.T))
    disp.plot_arrow(X,Y,Z, Ex, Ey, Ez, scalemode='lin', scale=5)
    p.show()
    

