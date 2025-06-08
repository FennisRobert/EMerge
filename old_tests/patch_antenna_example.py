import fem
import numpy as np


with fem.Simulation3D('PatchModel') as model:
    modeler = model.modeler

    boxes = modeler.box(0.1, 0.1, 0.1, (modeler.series(0, 0.15, 0.3, 0.45), 0, 0))
    
    model.preview()
    