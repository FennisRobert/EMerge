import fem
import numpy as np
import pyescher as pe

mm = 0.001
mil = 0.0254*mm

L0, L1, L2, L3 = 400, 660, 660, 660
W0, W1, W2, W3 = 50, 128, 8, 224
th = 31
er = 2.2
Ltot = L3 + 2*(L0+L1+L2)
Wtot = 424
Hair = 31
with fem.Simulation3D('Demo1_SIF', 'DEBUG') as m:

    pcbmat = fem.material.Material(2.2)
    pcbr = fem.modeling.PCBLayouter(Ltot, Wtot, th, Hair, (0, -Wtot/2, 0), unit=mil, material=pcbmat)

    pcbr.new(0,Wtot/2,W0, (1,0)).straight(L0, W0).straight(L1,W1).straight(L2,W2).straight(L3,W3)\
        .straight(L2,W2).straight(L1,W1).straight(L0,W0)
    
    p1 = fem.modeling.Plate((0,-Wtot*mil/2, -th*mil),
                            (0,Wtot*mil,0),
                            (0,0,Hair*mil+th*mil))
    p2 = fem.modeling.Plate((Ltot*mil,-Wtot*mil/2, -th*mil),
                            (0,Wtot*mil,0),
                            (0,0,Hair*mil+th*mil))
    
    polies = pcbr.compile_paths(0)

    
    pcb = pcbr.gen_pcb()
    
    air = pcbr.gen_air()
    air.material = pcbmat

    m.define_geometry(pcb, air, polies, p1, p2)

    m.physics.resolution = 0.1
    pcb.max_meshsize = (2*mm)
    m.mesher.set_boundary_size(polies[0].dimtags, 1*mm, max_size = 5*mm, edge_only=True)
    m.mesher.set_boundary_size(p1.dimtags, 2*mm, max_size=5*mm, edge_only=False)
    m.mesher.set_boundary_size(p2.dimtags, 2*mm, max_size=5*mm, edge_only=False)
    m.physics.set_frequency(np.linspace(0.2e9, 3e9, 31))

    m.generate_mesh('Demo1_Mesh.msh')

    m.preview()
    port1 = fem.bc.ModalPort(p1, 1, True)
    port2 = fem.bc.ModalPort(p2, 2, False)
    pec = fem.bc.PEC(polies[0])

    m.physics.assign(port1, port2, pec)

    m.physics.modal_analysis(port1, 1, True, TEM=True, freq=0.5e9)
    m.physics.modal_analysis(port2, 1, True, TEM=True, freq=0.5e9)

    from fem.plotting.pyvista import PVDisplay

    d = PVDisplay(m.mesh)
    d.add_object(pcb)
    d.add_object(polies, color='red')
    d.add_object(p1, color='blue', opacity=0.3)
    d.add_portmode(port1, port1.modes[0].k0, 21)
    d.add_portmode(port2, port2.modes[0].k0, 21)
    d.show()

    sol = m.physics.frequency_domain()
    
    f, S11 = sol.ax('freq').S(1,1)
    f, S21 = sol.ax('freq').S(2,1)
    pe.plot_lines(pe.Line(f/1e9, S11, label='S11'), pe.Line(f/1e9,S21,label='S21'), show_marker=True, 
                  transformation=pe.dB,
                  grid=True,
                  xlabel='Frequency (GHz)',
                  ylabel='S-parameters (dB)',
                  ylim=[-50,5],marker_size=1)


    