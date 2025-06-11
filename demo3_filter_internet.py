import fem
import numpy as np
import pyescher as pe
from fem.plotting.pyvista import PVDisplay

mm = 0.001
mil = 0.0254*mm
W = 20
D = 20
w0 = 37
l0 = 100
l1 = 314.22
l2 = 301.658
l3 = 300.589

w1, w2, w3, w4, w5, w6 = 18.8, 43.484, 44.331, 44.331, 43.484, 18.8
g1, g2, g3, g4, g5, g6 = 9.63, 24.84, 41.499, 41.499, 24.84, 9.63

th = 20
e = 0
Wtot = 2*l1 + 5*l2 + 7*e + 2*l0
WP = 200
Dtot = 750

extra = 100
with fem.Simulation3D('Demo3', PVDisplay, loglevel='DEBUG') as m:
    mat = fem.Material(3.55, color=(0.1,1.0,0.2), opacity=0.1)
    pcb = fem.modeling.PCBLayouter(th,4*th,unit=mil)

    pcb.new(0,140,w0,(1,0)).straight(l0).turn(0).straight(l1*0.8)\
        .straight(l1,w1, dy=abs(w1-w0)/2).jump(gap=g1, side='left', reverse=l1-e).straight(l1,w1)\
        .straight(l2,w2, dy=abs(w2-w1)/2).jump(gap=g2, side='left', reverse=l2-e).straight(l2,w2)\
        .straight(l3,w3).jump(gap=g3, side='left', reverse=l2-e).straight(l2,w3)\
        .straight(l3,w4).jump(gap=g4, side='left', reverse=l2-e).straight(l2,w4)\
        .straight(l2,w5).jump(gap=g5, side='left', reverse=l2-e).straight(l2,w5)\
        .straight(l1,w6, dy=abs(w2-w1)/2).jump(gap=g6, side='left', reverse=l1-e).straight(l1,w6)\
        .turn(0).straight(l1*0.8, w0).store('p2').straight(l0,w0, dy=abs(w1-w0)/2)
    
    pcb.new(l0,140,10,(0,-1)).straight(60)
    pcb.new(*pcb.stored_coords['p2'],10, (0,1)).straight(60)
    stripline = pcb.compile_paths(merge=True)
    
    pcb.determine_bounds(topmargin=150, bottommargin=150)

    diel = pcb.gen_pcb()
    air = pcb.gen_air()

    diel.material = mat

    p1 = pcb.wave_port(pcb.paths[0].start, width_multiplier=5)
    p2 = pcb.wave_port(pcb.paths[-3].end, width_multiplier=5)

    m.physics.resolution = 0.2

    m.physics.set_frequency(np.linspace(5.2e9,6.2e9,3))

    m.define_geometry(stripline, diel, p1, p2, air)

    m.mesher.set_boundary_size(stripline.dimtags, 0.7*mm, edge_only=False, max_size=10*mm, growth_distance=10)

    m.generate_mesh('pcbmesh.msh')

    m.view()
    port1 = fem.bc.ModalPort(p1, 1, True)
    port2 = fem.bc.ModalPort(p2, 2, False)
    pec = fem.bc.PEC(stripline)

    m.physics.assign(port1, port2, pec)
    

    m.physics.modal_analysis(port1, 1, direct=True, TEM=True, freq=10e9)
    m.physics.modal_analysis(port2, 1, direct=True, TEM=True, freq=10e9)

    d = PVDisplay(m.mesh)
    d.add_object(diel, color='green', opacity=0.5)
    d.add_object(stripline, color='red')
    d.add_object(p1, color='blue', opacity=0.3)
    d.add_portmode(port1, port1.modes[0].k0, 21)
    d.add_portmode(port2, port2.modes[0].k0, 21)
    d.show()

    data = m.physics.frequency_domain()

    

    d = PVDisplay(m.mesh)
    d.add_object(diel, color='green', opacity=0.5)
    d.add_object(stripline, color='red')
    d.add_object(p1, color='blue', opacity=0.3)
    d.add_portmode(port1, port1.modes[0].k0, 21)
    d.add_portmode(port2, port2.modes[0].k0, 21)
    d.show()
    
    f, S11 = data.ax('freq').S(1,1)
    f, S21 = data.ax('freq').S(2,1)
    pe.plot_lines(pe.Line(f/1e9, S11, label='S11'), pe.Line(f/1e9,S21,label='S21'), show_marker=True, 
                  transformation=pe.dB,
                  grid=True,
                  xlabel='Frequency (GHz)',
                  ylabel='S-parameters (dB)',
                  ylim=[-50,5],marker_size=1)

    pe.plot_lines(pe.Line(S11.real, S11.imag), xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), grid=True)
