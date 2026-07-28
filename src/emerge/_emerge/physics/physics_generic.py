from .._global import _GlobalHandler
from emsutil import Material
from ..geometry import GeoSurface, GeoVolume
from collections import defaultdict
from loguru import logger
from .material_assignment import MaterialAssignment
import numpy as np

class SimulationError(Exception):
    pass


class GenericPhysics3D:

    def _generate_material_assignment(self):
            """Retrieve the material properties of the geometry"""
    
            # In order to make EMerge projects saveable, the Materials are told which
            # geometries they have been assigned to. These material lists are stored in the final solution
            # The reason is that per simulation and frequency, the material propery value may be different.
            if self.mat_assy is not None:
                logger.debug('   Using cached material assignment.')
                return
            self.mat_assy = MaterialAssignment(self._state.current_geo_state)
            self.mat_assy.set_tet_assignment(self.mesh._get_tet_to_tag(), self.mesh.centers)
    