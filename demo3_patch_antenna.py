import fem
import pyescher as pe
import numpy as np
from fem.plotting.pyvista import PVDisplay
import matplotlib.pyplot as plt

mm = 0.001

np.set_printoptions(precision=3, suppress=True)

margin = 5*mm
Nmodes = 1
f1 = 0.5e9
f2 = 3e9

Wpatch = 53*mm
Lpatch = 52*mm
wline = 3.2*mm
wstub = 7*mm
lstub = 15.5*mm
wsub = 100*mm
hsub = 100*mm
th = 1.524*mm

Hair = 40*mm

er = 3.38

f1 = 1.54e9
f2 = 1.6e9

with fem.Simulation3D('MySimulation', PVDisplay, loglevel='DEBUG') as model:
    dielectric = fem.modeling.Box(wsub, hsub, th, position=(-wsub/2, -hsub/2, -th))

    air = fem.modeling.SidedBox(wsub, hsub, Hair, position=(-wsub/2, -hsub/2, 0))
    
    rpatch = fem.modeling.XYPlate(Wpatch, Lpatch, position=(-Wpatch/2, -Lpatch/2, 0))
    
    cutout1 = fem.modeling.XYPlate(wstub, lstub, position=(-wline/2-wstub, -Lpatch/2, 0))
    cutout2 = fem.modeling.XYPlate(wstub, lstub, position=(wline/2, -Lpatch/2, 0))

    line = fem.modeling.XYPlate(wline, lstub, position=(-wline/2, -Lpatch/2, 0))

    port = fem.modeling.Plate(np.array([-wline/2, -Lpatch/2, -th]), np.array([wline, 0, 0]), np.array([0, 0, th]))

    rpatch = fem.modeling.remove(rpatch, cutout1)
    rpatch = fem.modeling.remove(rpatch, cutout2)
    rpatch = fem.modeling.add(rpatch, line)
    
    rpatch.material = fem.COPPER # Only for viewing

    dielectric.material = fem.Material(er, color=(0.0, 0.5, 0.0), opacity=0.6)

    model.physics.resolution = 0.25
    
    model.physics.set_frequency(1.6e9)

    model.define_geometry([dielectric, air, rpatch, port])

    model.mesher.set_boundary_size(rpatch.dimtags, 5*mm, edge_only=True, max_size=20*mm)
    model.mesher.set_boundary_size(model.select.face.near(0.0,0.0,-th).dimtags, 8*mm, max_size=50*mm)

    model.generate_mesh()

    #model.view()

    xyzs = rpatch.select.sample(50)
    
    port = fem.bc.LumpedPort(port, 1, width=wline, height=th, direction=fem.ZAX, active=True, Z0=50)

    boundary_selection = model.select.face.inlayer(0, 0, 2*th, np.array([0, 0, Hair]))
    
    abc = fem.bc.AbsorbingBoundary(boundary_selection)
    
    pec = fem.bc.PEC(rpatch)

    model.physics.assign(port, pec, abc)

    model.physics.set_frequency(np.linspace(f1, f2, 3))

    model.physics.solveroutine.use_direct = False
    model.physics.solveroutine.use_preconditioner = True
    data = model.physics.frequency_domain()
    
    xs, ys, zs = fem.YAX.pair(fem.ZAX).span(wsub, Hair, 31, (0, -wsub/2, -th))

    freqs = np.array(model.physics.frequencies)

    freqs, S11 = data.ax('freq').S(1,1)
    
    pe.plot_lines(pe.Line(freqs/1e9, S11, label='S11', transformation=pe.dB), 
                     xlabel='Frequency (GHz)', ylabel='S-parameter (dB)',
                     title='S-parameters', grid=True, show_marker=True)
    
    topsurf = model.mesh.boundary_surface(boundary_selection.tags, (0,0,0))

    Ein, Hin = data.item(0).interpolate(*topsurf.exyz).EH
    
    theta = np.linspace(-np.pi, 1*np.pi, 201)
    phi = 0*theta
    E, H = fem.physics.edm.stratton_chu(Ein, Hin, topsurf, theta, phi, data.item(0).k0)
    
    pe.plot_lines(pe.Line(theta, fem.norm(E)), transformation=pe.dB)
    
    ### Create a polar farfield plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')

    ax.plot(theta, fem.norm(E), label='E-field', color='blue')
    plt.show()

