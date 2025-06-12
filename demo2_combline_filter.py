import fem
import numpy as np
import pyescher as pe
from fem.plotting.pyvista import PVDisplay
mm = 0.001
mil = 0.0254*mm

a = 240*mil
b = 248*mil
d1 = 10*mil
d2 = 10*mil
dc = 8.5*mil
lr1 = b-d1
lr2 = b-d2
W = 84*mil
S1 = 117*mil
S2 = 136*mil
C1 = b-dc
h = 74*mil
wi = 84*mil
Lbox = 5*W + 2*(S1+S2+wi)

x1 = wi+W/2
x2 = x1 + W + S1
x3 = x2 + W + S2
x4 = x3 + W + S2
x5 = x4 + W + S1

rout = 81*mil
rin = 25*mil
lfeed = 100*mil
with fem.Simulation3D('Combline_DEMO', PVDisplay, loglevel='DEBUG') as m:
    box = fem.modeling.Box(Lbox, a, b, position=(0,-a/2,0))
    stubs = m.modeler.cyllinder(W/2, m.modeler.series(C1, lr1, lr2, lr1, C1), position=(m.modeler.series(x1, x2, x3, x4, x5), 0, 0), NPoly=10)

    feed1out = fem.modeling.Cyllinder(rout, lfeed, fem.CoordinateSystem(fem.ZAX, fem.YAX, fem.XAX, np.array([-lfeed, 0, h])), Nsections=12)
    feed1in = fem.modeling.Cyllinder(rin, lfeed+wi+W/2, fem.CoordinateSystem(fem.ZAX, fem.YAX, fem.XAX, np.array([-lfeed, 0, h])), Nsections=8)
    feed2out = fem.modeling.Cyllinder(rout, lfeed, fem.CoordinateSystem(fem.ZAX, fem.YAX, fem.XAX, np.array([Lbox, 0, h])), Nsections=12)
    feed2in = fem.modeling.Cyllinder(rin, lfeed+wi+W/2, fem.CoordinateSystem(fem.ZAX, fem.YAX, fem.XAX, np.array([Lbox-wi-W/2, 0, h])), Nsections=8)
    
    for ro in stubs:
        box = fem.modeling.subtract(box, ro)
    
    box = fem.modeling.subtract(box, feed1in, remove_tool=False)
    box = fem.modeling.subtract(box, feed2in, remove_tool=False)
    feed1out = fem.modeling.subtract(feed1out, feed1in, remove_tool=True)
    feed2out = fem.modeling.subtract(feed2out, feed2in, remove_tool=True)
    
    m.define_geometry(box, feed1out, feed2out)

    m.view()

    m.physics.set_frequency(np.linspace(6e9, 8e9, 41))
    m.physics.resolution = 0.04
    m.generate_mesh()

    m.view()

    port1 = fem.bc.ModalPort(m.select.face.near(-lfeed, 0, h), 1, True)
    port2 = fem.bc.ModalPort(m.select.face.near(Lbox+lfeed, 0, h), 2, False)

    m.physics.assign(port1, port2)

    m.physics.modal_analysis(port1, 1, direct=True, TEM=True)
    m.physics.modal_analysis(port2, 1, direct=True, TEM=True)

    from fem.plotting.pyvista import PVDisplay
    
    data = m.physics.frequency_domain()

    f, S11 = data.ax('freq').S(1,1)
    f, S21 = data.ax('freq').S(2,1)

    xs = np.linspace(0, Lbox, 41)
    ys = np.linspace(-a/2, a/2, 11)
    zs = np.linspace(0, b, 15)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    Ex, Ey, Ez = data.item(11).interpolate(X,Y,Z,data.item(11).freq).E

    pe.plot_lines(pe.Line(f/1e9, S11, label='S11'), pe.Line(f/1e9,S21,label='S21'), show_marker=True, 
                  transformation=pe.dB,
                  grid=True,
                  xlabel='Frequency (GHz)',
                  ylabel='S-parameters (dB)',
                  ylim=[-50,5],marker_size=2)
    pe.plot_lines(pe.Line(S11.real, S11.imag), xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))

    d = PVDisplay(m.mesh)
    d.add_object(box, opacity=0.1, show_edges=True)
    d.add_quiver(X,Y,Z, Ex.real, Ey.real, Ez.real)
    d.add_object(feed1out, opacity=0.1)
    d.add_portmode(port1, port1.modes[0].k0, 21)
    d.add_portmode(port2, port2.modes[0].k0, 21)
    d.show()