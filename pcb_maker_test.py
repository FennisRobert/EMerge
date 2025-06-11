import fem
import numpy as np
import pyescher as pe

with fem.Simulation3D('PCB Making test', loglevel='DEBUG') as m:

    pcb = fem.modeling.PCBLayouter(1, 5, 0.001)

    path = pcb.new(0,5,2,(1,0)).straight(5).straight(5,2).straight(5,2)

    lp1 = pcb.lumped_port(path.start)

    lines = pcb.compile_paths(True)

    pcb.determine_bounds(5,5,0,5)

    lp2 = pcb.wave_port(path.end, 4)

    diel = pcb.gen_pcb()
    air = pcb.gen_air()

    m.define_geometry(lines, lp1, diel, lp2, air)

    m.physics.set_frequency(np.linspace(4e9, 6e9, 11))
    m.physics.resolution = 0.10
    m.generate_mesh()

    p1 = fem.bc.LumpedPort(lp1, 1, 0.002, 0.001, fem.ZAX, True, Z0=72)
    p2 = fem.bc.ModalPort(lp2, 2, False)
    
    
    pec = fem.bc.PEC(lines)
    m.physics.assign(p1, p2, pec)

    m.physics.modal_analysis(p2, 1, True, TEM=True)

    data = m.physics.frequency_domain()

    f, S11 = data.ax('freq').S(1,1)
    f, S21 = data.ax('freq').S(2,1)

    pe.plot_lines(pe.Line(f/1e9, S11, label='S11'), pe.Line(f/1e9, S21, label='S21'), transformation=pe.dB, grid=True, show_marker=True)

    from fem.plotting.pyvista import PVDisplay

    d = PVDisplay(m.mesh)
    d.add_portmode(p2, data.item(10).k0, 15, dv=(0.1e-3,0,0))
    d.add_object(diel, opacity=0.3, color='green')
    d.add_object(lines, color='red')
    d.show()