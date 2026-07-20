from .._global import _GlobalHandler
from emsutil import Material
from ..geometry import GeoSurface, GeoVolume
from collections import defaultdict
from loguru import logger

import numpy as np

class SimulationError(Exception):
    pass


class GenericPhysics3D:

    def _get_material_assignment(self, volumes: list[GeoVolume]) -> list[Material]:
        '''Retrieve the material properties of the geometry'''
        
        # In order to make EMerge projects saveable, the Materials are told which
        # geometries they have been assigned to. These material lists are stored in the final solution
        # The reason is that per simulation and frequency, the material propery value may be different.

        # Reset index assingments
        for vol in volumes:
            vol.material.reset()
        
        # collect all materials
        materials = []
        assignment_dict: dict[int, list[GeoVolume]] = defaultdict(list)
        i = 0
        for vol in volumes:
            for tag in vol.tags:
                assignment_dict[tag].append(vol)
            if vol.material not in materials:
                materials.append(vol.material)
                vol.material._hash_key = i
                i += 1
        
        # Check competing priorities!
        for domaintag, volumelist in assignment_dict.items():
            priolist = [vol._priority for vol in volumelist]
            maxprio = max(priolist)
            if priolist.count(maxprio) > 1:
                vols = [vol for vol in volumelist if vol._priority==maxprio]
                logger.warning(f'Domain with tag {domaintag} has multiple geometries imposing a material to them: {vols}. Consider setting priorities to decide which volume is more important.')
                _GlobalHandler.active().debugcollector.add_report(f'Domain with tag {domaintag} has multiple geometries imposing a material to them: {vols}. Consider setting priorities to decide which volume is more important.')
            
        xs = self.mesh.centers[0,:]
        ys = self.mesh.centers[1,:]
        zs = self.mesh.centers[2,:]
        
        matassign = -1*np.ones((self.mesh.n_tets,), dtype=np.int64)
        
        for volume in sorted(volumes, key=lambda x: x._priority):
        
            for dimtag in volume.dimtags:
                
                tet_ids = self.mesh.get_tetrahedra(dimtag[1])
                
                matassign[tet_ids] = volume.material._hash_key
        
        if np.any(matassign==-1):
            raise SimulationError(f'Tetrahedra detected with unassigned materials: {np.argwhere(matassign==-1)}')
        
        for mat in materials:
            ids = np.argwhere(matassign==mat._hash_key).flatten()
            mat.initialize(xs[ids], ys[ids], zs[ids], ids)
                    
        
        return materials