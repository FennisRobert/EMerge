from __future__ import annotations
from emsutil import Material
from ..geometry import GeoObject
import numpy as np
from loguru import logger

class MaterialAssignment:
    
    def __init__(self, geos: list[GeoObject]):
        
        self._geos: list[GeoObject] = geos
        self.mat2ind: dict[Material, int] = dict()
        self.ind2mat: dict[int, Material] = dict()
        self.materials: list[Material] = list()
        self.geomat: dict[tuple[int,int], Material] = dict()
        
        self.obj_assignment: dict[int, dict[int, int]] = {
            0: dict(),
            1: dict(),
            2: dict(),
            3: dict(),
        }
        
        self.tet_to_matid: np.ndarray | None = None
        self.tri_to_matid: np.ndarray | None = None

        self._parse_assignment()
        
    def _parse_assignment(self) -> None:
        
        # Step 1: Generate unique list of materials
        ctr = 0
        names = []
        for mat in [geo.material for geo in self._geos if geo.material is not None]:
            if mat.name not in names:
                self.mat2ind[mat] = ctr
                self.ind2mat[ctr] = mat
                logger.debug(f'   {ctr}: {mat.name}')
                ctr += 1
                names.append(mat.name)
                self.materials.append(mat)
        # From here on out each material has a unique index
        # Step 2: Assign by priority
        for geo in sorted(self._geos, key=lambda x: x._priority):
            # Only consider geos with a material assigned
            if geo.material is None:
                continue

            for dimtag in geo.dimtags:
                self.geomat[dimtag] = self.mat2ind[geo.material]
        
        for (d,t), matid in self.geomat.items():
            self.obj_assignment[d][t] = matid
    
    def get_material(self, dim: int, tag: int) -> Material | None:
        return self.ind2mat[self.obj_assignment[dim][tag]]
    
    def set_tet_assignment(self, tet_to_tag: np.ndarray) -> None:
        self.tet_to_matid = np.zeros_like(tet_to_tag, dtype=np.int64) - 1
        for tag, matid in self.obj_assignment[3].items():
            mask = np.argwhere(tet_to_tag==tag)
            self.tet_to_matid[mask] = matid

        if np.any(self.tet_to_matid == -1):
            raise RuntimeError(
                f"Tetrahedra detected with unassigned materials: {np.argwhere(self.tet_to_matid == -1)}"
            )
            
    def frequency_dependent(self) -> bool:
        """Returns True if there are frequency dependent materials

        Returns:
            bool: _description_
        """
        for mat in self.materials:
            if mat.frequency_dependent:
                return True
        return False