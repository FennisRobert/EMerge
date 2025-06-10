import fem
import pyescher as pe
from fem.plotting.pyvista import PVDisplay
import numpy as np

with fem.Simulation3D('SelectionTest',PVDisplay) as m:

    box = fem.modeling.SiddBox(0.02286, 0.05, 0.01016)

    m.define_geometry(box)

    m.physics.set_frequency(np.linspace(9e9,10e9,5))
    m.physics.resolution = 0.2
    m.generate_mesh()

    port1 = fem.bc.RectangularWaveguide(box.front, 1, True)
    port2 = fem.bc.RectangularWaveguide(box.back, 2, False)

    m.physics.assign(port1, port2)

    #m.physics.modal_analysis(port1, 1, True, False)
    #m.physics.modal_analysis(port2, 2, True, False)
    
    data = m.physics.frequency_domain()

    freq, S21 = data.ax('freq').S(2,1)
    freq, S11 = data.ax('freq').S(1,1)

    pe.plot_lines(pe.Line(freq, S11, label='S11'), pe.Line(freq, S21, label='S21'), transformation=pe.dB)

    X,Y,Z = box.bottom.sample(20)

    Ex, Ey, Ez = data.item(1).interpolate(X,Y,Z).E

    m.display.add_object(box, opacity=0.1)
    m.display.add_quiver(X.flatten(), Y.flatten(), Z.flatten(), Ex.real.flatten(), Ey.real.flatten(), Ez.real.flatten())
    m.display.show()