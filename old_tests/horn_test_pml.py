
import fem
from scipy.spatial import Delaunay
import pyescher as pe
import numpy as np


mm = 0.001

viewer = pe.Viewer()

W = 100*mm
D = 100*mm
H = 15*mm

f1 = 8e9
f2 = 9e9

wga = 22.86*mm
wgb = 10.16*mm
wgL = 30*mm
thpml = 20*mm
inch = 25.4*mm

LH = 4.3*inch
AP1 = (wga, wgb)
AP2 = (2.51*inch, 1.960*inch)

with fem.Simulation3D('MySimulation') as model:
    model.physics.set_order(2)
    boxes = fem.modeling.pmlbox(W, D, H, (-W/2, -D/2, 0), thickness= thpml, Nlayers=1, 
                                top=True, front=True, back=True,
                                left=True, right=True, bottom=False)

    wgfeed = fem.geo3d.Box(wga, wgb, wgL, (-wga/2, -wgb/2, -wgL-LH))
    horn = fem.modeling.Horn(AP1, AP2, LH, fem.CoordinateSystem(fem.XAX, fem.YAX, fem.ZAX, origin=(0,0,-LH)))
    model.define_geometry(boxes + [wgfeed, horn])
    model.physics.resolution = 0.2
    model.physics.set_frequency(f1)

    model.generate_mesh()

    air = boxes[0]
    
    #model.mesh.plot_gmsh()

    port1 = fem.bc.RectangularWaveguide(
        model.select.face.near(-wga/2, -wgb/2, -wgL-LH), 1, True)

    model.physics.assign(port1)

    model.physics.set_frequency(np.linspace(f1, f2, 2))

    data = model.run_frequency_domain()

    S11 = data.glob('S11')[:]

    xs, ys, zs = fem.YAX.pair(fem.ZAX).grid(np.linspace(0,D+2*thpml,101), 
                                            np.linspace(0,wgL+H+thpml+LH,101), (0,-D/2-thpml,-wgL-LH))

    Ex, Ey, Ez = data.item(0).interpolate(xs, ys, zs).E

    airfacetags = fem.FaceSelection(air.boundary()).exclude(lambda x, y, z: z > 0).tags

    topsurf = model.mesh.boundary_surface(airfacetags, (0,0,0))

    Ein, Hin = data.item(0).interpolate(*topsurf.exyz).EH
    
    theta = np.linspace(0, 1*np.pi, 201)
    phi = 0*theta
    E, H = fem.physics.edm.stratton_chu(Ein, Hin, topsurf, theta, phi, data.item(0).glob('k0')[0])
    pe.plot_lines(pe.Line(theta, fem.norm(E)), transformation=pe.dB)

    with viewer.new3d('E-field') as v:
        #v.mesh(*model.get_boundary(horn))
        v.mesh(*model.get_boundary(tags = airfacetags))
        v.surf(xs, ys, zs, scalars=np.abs(Ey))

                                            