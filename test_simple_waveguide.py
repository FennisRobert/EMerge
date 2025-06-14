import fem
import pyescher as pe
import numpy as np

mm = 0.001

a = 22.86*mm
b = 10.16*mm
L = 50*mm

with fem.Simulation3D('Waveguide', loglevel='DEBUG') as m:
    wg = fem.modeling.SidedBox(a, L, b)
    cutout = fem.modeling.Box(a, 2*mm, b/2, position=(0,L/2,0))
    wg = fem.modeling.remove(wg, cutout)
    m.define_geometry(wg)

    m.physics.set_frequency(np.linspace(8e9, 10e9, 31))

    m.generate_mesh()

    m.view()
    #mp1 = fem.bc.ModalPort(wg.face('front'), 1)
    #mp2 = fem.bc.ModalPort(wg.face('back'), 2)

    # m.physics.assign(mp1, mp2)

    # m.physics.modal_analysis(mp1, 1)
    # m.physics.modal_analysis(mp2, 2)

    # data = m.physics.frequency_domain()

    # freq, S11 = data.ax('freq').S(1,1)
    # freq, S21 = data.ax('freq').S(2,1)

    # pe.plot_lines(
    #     pe.Line(freq/1e9, S11, label='S11'),
    #     pe.Line(freq/1e9, S21, label='S21'),
    #     transformation=pe.dB
    # )

