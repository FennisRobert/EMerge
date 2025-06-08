import fem
from fem.modeling.modeler import Modeler
import numpy as np

with fem.Simulation3D('TestSim') as m:
    M = Modeler()
    plane = fem.modeling.Plate(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]))
    
    
    boxes = M.rotated((0,0,0), (0,1,0), M.series(45, 135))\
        .mirror((0,0,0),(0,0,1)).box(M.series(0.1,0.12), 0.1, 0.1, (0.2, 0,0))

    m.define_geometry(boxes + [plane,])

    m.mesh.plot_gmsh()