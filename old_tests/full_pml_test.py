
import fem
from scipy.spatial import Delaunay
import pyescher as pe
import numpy as np


mm = 0.001

#viewer = pe.Viewer()

W = 50*mm
D = 60*mm
H = 70*mm

f1 = 8e9
f2 = 9e9

wga = 22.86*mm
wgb = 10.16*mm
wgL = 30*mm
thpml = 20*mm
inch = 25.4*mm

LH = 4.3*inch
AP1 = (wga, wgb)
AP2 = (1.960*inch, 2.51*inch)

with fem.Simulation3D('MySimulation') as model:
    model.physics.set_order(2)
    boxes = fem.modeling.pmlbox(W, D, H, (-W/2, -D/2, 0), 
                                thickness= thpml, 
                                Nlayers=2,
                                top=True,
                                bottom=True,
                                right=True,
                                left=True,
                                front=True,
                                back=True)

    model.physics.set_frequency(np.linspace(f1, f2, 201))

    model.define_geometry(boxes)
    model.generate_mesh()
    model.mesh.plot_gmsh()