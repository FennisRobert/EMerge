import fem
import numpy as np
import pyvista as pv

mm = 0.001
wga = 25*mm#22.86*mm
wgb = 10.16*mm
L = 50*mm

f1 = 8e9
f2 = 9e9
freqs = np.linspace(f1, f2, 21)
dist = 25*mm
with fem.Simulation3D('Waveguide') as m:
    wg = fem.modeling.Box(wga, L, wgb, position=(-wga/2, 0, 0))
    
    cutout = fem.modeling.Box(wga, 1*mm, wgb/2, position=(-wga/2, L/2-dist/2, 0))
    cutout2 = fem.modeling.Box(wga, 1*mm, wgb/2, position=(-wga/2, L/2+dist/2, 0))
    wgout = fem.modeling.remove(wg, cutout)
    wgout = fem.modeling.remove(wg, cutout2)

    m.physics.set_frequency(freqs)
    m.physics.resolution = 0.3
    m.define_geometry(wg)

    m.generate_mesh()

    fp1 = wg.front
    fp2 = wg.back

    port1 = fem.bc.RectangularWaveguide(fp1, 1, True)
    port2 = fem.bc.RectangularWaveguide(fp2, 2, False)

    m.physics.assign(port1, port2)

    data = m.physics.frequency_domain()

    f, S21 = data.ax('freq').S(2,1)
    f, S11 = data.ax('freq').S(1,1)

    #pe.plot_lines(pe.Line(f, S21), pe.Line(f,S11), transformation=pe.dB)

    
    X, Y, Z = fem.XYPLANE.grid(np.linspace(-wga/2,wga/2,21), 
                               np.linspace(0, L, 41), 
                               np.array([0,0,wgb/2]))
    
    fieldata = data.item(2)
    E, H = fieldata.interpolate(X, Y, Z).EH

    from fem.plotting.pyvista import Display

    p = pv.Plotter()

    d = Display(m.mesh, p)

    p.add_mesh(d.mesh(wgout), color='lightblue', opacity=0.1)
    d.plot_portmode(port1, data.item(2).k0)
    d.plot_portmode(port2, data.item(2).k0)
    d.plot_arrow(X,Y,Z, E[0,:], E[1,:], E[2,:], scale=2)
    p.show()