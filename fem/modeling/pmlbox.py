# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

from ..geo3d import GMSHVolume
from .shapes import Box, Sphere, Alignment, Plate
from ..material import Material, AIR
import gmsh
import numpy as np
from functools import partial


def _add_pml_layer(center: tuple[float, float, float],
                   dims: tuple[float, float, float],
                   direction: tuple[float, float, float],
                   thickness: float,
                   Nlayers: int,
                   exponent: float,
                   deltamax: float,
                   material: Material) -> list[GMSHVolume]:
    px, py, pz = center
    W,D,H = dims
    dx, dy, dz = direction

    pml_block_size = [W, D, H]
    new_center = [px, py, pz]
    
    sxf = lambda x, y, z: np.ones_like(x, dtype=np.complex128)
    syf = lambda x, y, z: np.ones_like(x, dtype=np.complex128)
    szf = lambda x, y, z: np.ones_like(x, dtype=np.complex128)
    
    p0x = px + dx*(W/2+thickness/2)
    p0y = py + dy*(D/2+thickness/2)
    p0z = pz + dz*(H/2+thickness/2)

    tW, tD, tH = W, D, H

    if dx != 0:
        tW = thickness
        pml_block_size[0] = thickness
        new_center[0] = new_center[0] + (W/2 + thickness/2) * dx
        def sxf(x, y, z):
            return 1 - 1j * (dx*(x-px-(dx*W/2)) / thickness) ** exponent * deltamax
    if dy != 0:
        tD = thickness
        pml_block_size[1] = thickness
        new_center[1] = new_center[1] + (D/2 + thickness/2) * dy
        def syf(x, y, z):
            return 1 - 1j * (dy*(y-py-(dy*D/2)) / thickness) ** exponent * deltamax
    if dz != 0:
        tH = thickness
        pml_block_size[2] = thickness
        new_center[2] = new_center[2] + (H/2 + thickness/2) * dz
        def szf(x, y, z):
            return 1 - 1j * (dz*(z-pz-(dz*H/2)) / thickness) ** exponent * deltamax
    
    if Nlayers == 1:
        def sxf(x, y, z):
            return (1 - 1.1j)*np.ones_like(x)
        def syf(x, y, z):
            return (1 - 1.1j)*np.ones_like(x)
        def szf(x, y, z):
            return (1 - 1.1j)*np.ones_like(x)

    def ermat(x, y, z):
        ers = np.zeros((3,3,x.shape[0]), dtype=np.complex128)
        ers[0,0,:] = material.er * syf(x,y,z)*szf(x,y,z)/sxf(x,y,z)
        ers[1,1,:] = material.er * szf(x,y,z)*sxf(x,y,z)/syf(x,y,z)
        ers[2,2,:] = material.er * sxf(x,y,z)*syf(x,y,z)/szf(x,y,z)
        return ers
    def urmat(x, y, z):
        urs = np.zeros((3,3,x.shape[0]), dtype=np.complex128)
        urs[0,0,:] = material.ur * syf(x,y,z)*szf(x,y,z)/sxf(x,y,z)
        urs[1,1,:] = material.ur * szf(x,y,z)*sxf(x,y,z)/syf(x,y,z)
        urs[2,2,:] = material.ur * sxf(x,y,z)*syf(x,y,z)/szf(x,y,z)
        return urs
    
    pml_box = Box(*pml_block_size, new_center, Alignment.CENTER)
    pml_box._unset_constraints = True
    
    planes = []
    thl = thickness / Nlayers
    planes = []

    if dx != 0:
        ax1 = np.array([0, tD, 0])
        ax2 = np.array([0, 0, tH])
        for n in range(Nlayers-1):
            plate = Plate(np.array([p0x-dx*thickness/2 + dx*(n+1)*thl, p0y-tD/2, p0z-tH/2]), ax1, ax2)
            planes.append(plate)
    if dy != 0:
        ax1 = np.array([tW, 0, 0])
        ax2 = np.array([0, 0, tH])
        for n in range(Nlayers-1):
            plate = Plate(np.array([p0x-tW/2, p0y-dy*thickness/2 + dy*(n+1)*thl, p0z-tH/2]), ax1, ax2)
            planes.append(plate)
    if dz != 0:
        ax1 = np.array([tW, 0, 0])
        ax2 = np.array([0, tD, 0])
        for n in range(Nlayers-1):
            plate = Plate(np.array([p0x-tW/2, p0y-tD/2, p0z-dz*thickness/2 + dz*(n+1)*thl]), ax1, ax2)
            planes.append(plate)
    
    pml_box.material = Material(_neff=np.sqrt(material.er*material.ur), _fer=ermat, _fur=urmat)
    pml_box.max_meshsize = thickness/Nlayers * 2
    pml_box._embeddings = planes
    
    return pml_box


def pmlbox(width: float,
            depth: float,
            height: float,
            position: tuple = (0, 0, 0),
            alignment: Alignment = Alignment.CORNER,
            material: Material = AIR,
            thickness: float = 0.1,
            Nlayers: int = 1,
            exponent: float = 2.0,
            deltamax: float = 8.0,
            top: bool = True,
            bottom: bool = False,
            left: bool = False,
            right: bool = False,
            front: bool = False,
            back: bool = False) -> list[GMSHVolume]:
    
    px, py, pz = position
    if alignment == Alignment.CORNER:
        px = px + width / 2
        py = py + depth / 2
        pz = pz + height / 2

    position = (px, py, pz)
    main_box = Box(width, depth, height, position, Alignment.CENTER)
    main_box.material = material

    main_box._unset_constraints = True
    other_boxes = []

    addpml = partial(_add_pml_layer, center=(px, py, pz), dims=(width, depth, height),
                     thickness=thickness, Nlayers=Nlayers,
                     exponent=exponent, deltamax=deltamax, material=material)
    xs = [0,]
    ys = [0,]
    zs = [0,]
    if top:
        zs.append(1)
    if bottom:
        zs.append(-1)
    if left:
        xs.append(-1)
    if right:
        xs.append(1)
    if front:
        ys.append(-1)
    if back:
        ys.append(1)
    for x in xs:
        for y in ys:
            for z in zs:
                if x == 0 and y == 0 and z == 0:
                    continue
                box = addpml(direction=(x, y, z))
                other_boxes.append(box
                    
                )
                #planes.extend(planes)
    
    return [main_box] + other_boxes