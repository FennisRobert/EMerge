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

# Last Cleanup: 2025-03-12
from ..geometry import GeoVolume
from .shapes import Box, Alignment, Plate
from emsutil import Material, AIR, CoordDependent
import numpy as np
from functools import partial

def _sgn(num: float) -> str:
    if num > 0:
        return '+'
    elif num < 0:
        return '-'
    return ''

def _add_pml_layer(center: tuple[float, float, float],
                   dims: tuple[float, float, float],
                   direction: tuple[float, float, float],
                   thickness: float,
                   Nlayers: int,
                   N_mesh_layers: int,
                   exponent: float,
                   deltamax: float,
                   material: Material) -> GeoVolume:
    
    if material.frequency_dependent:
        raise ValueError('EMerge cannot handle frequency dependent material properties for PML layers at this point.')
    if material.coordinate_dependent:
        raise ValueError('Its not possible to define PML regions for materials that are coordinate dependent.')
    
    mater = material.er.scalar(1e9)
    matur = material.ur.scalar(1e9)
    
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

    _namesuffix = ''
    if dx != 0:
        tW = thickness
        pml_block_size[0] = thickness
        new_center[0] = new_center[0] + (W/2 + thickness/2) * dx
        def sxf(x, y, z):
            return 1 - 1j * (dx*(x-px-(dx*W/2)) / thickness) ** exponent * deltamax
        _namesuffix += f'{_sgn(dx)}x'
    if dy != 0:
        tD = thickness
        pml_block_size[1] = thickness
        new_center[1] = new_center[1] + (D/2 + thickness/2) * dy
        def syf(x, y, z):
            return 1 - 1j * (dy*(y-py-(dy*D/2)) / thickness) ** exponent * deltamax
        _namesuffix += f'{_sgn(dy)}y'
    if dz != 0:
        tH = thickness
        pml_block_size[2] = thickness
        new_center[2] = new_center[2] + (H/2 + thickness/2) * dz
        def szf(x, y, z):
            return 1 - 1j * (dz*(z-pz-(dz*H/2)) / thickness) ** exponent * deltamax
        _namesuffix += f'{_sgn(dz)}z'
    def ermat(x, y, z):
        ers = np.zeros((3,3,x.shape[0]), dtype=np.complex128)
        ers[0,0,:] = mater * syf(x,y,z)*szf(x,y,z)/sxf(x,y,z)
        ers[1,1,:] = mater * szf(x,y,z)*sxf(x,y,z)/syf(x,y,z)
        ers[2,2,:] = mater * sxf(x,y,z)*syf(x,y,z)/szf(x,y,z)
        return ers
    
    def urmat(x, y, z):
        urs = np.zeros((3,3,x.shape[0]), dtype=np.complex128)
        urs[0,0,:] = matur * syf(x,y,z)*szf(x,y,z)/sxf(x,y,z)
        urs[1,1,:] = matur * szf(x,y,z)*sxf(x,y,z)/syf(x,y,z)
        urs[2,2,:] = matur * sxf(x,y,z)*syf(x,y,z)/szf(x,y,z)
        return urs
    
    pml_box = Box(*pml_block_size, new_center, alignment=Alignment.CENTER)
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
    
    erfunc = CoordDependent(max_value=mater, matrix=ermat)
    urfunc = CoordDependent(max_value=matur, matrix=urmat)

    pml_box.properties += Material(er=erfunc, ur=urfunc,_neff=np.sqrt(mater*matur), color='#bbbbff', opacity=0.1, name=f'PML_mat[{_namesuffix}]')
    pml_box.max_meshsize = thickness/N_mesh_layers
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
            N_mesh_layers: int = 5,
            exponent: float = 1.5,
            deltamax: float = 8.0,
            sides: str = '',
            top: bool = False,
            bottom: bool = False,
            left: bool = False,
            right: bool = False,
            front: bool = False,
            back: bool = False) -> list[GeoVolume]:
    """Generate a block of uniform material (default air) with optional PML boxes around it

    This constructor uses coordinate-dependent material properties so only 1 layer is needed for the PML box. As a standin,
    the mesh discretization will be based on the thickness/number of mesh layers. If the PML layer is over-meshsed, try decreasing 
    the number of mesh layers.


    Args:
        width (float): The width of the box
        depth (float): The depth of the box
        height (float): The height of the box
        position (tuple, optional): The placmeent of the box. Defaults to (0, 0, 0).
        alignment (Alignment, optional): Which point of the box is placed at the given coordinate. Defaults to Alignment.CORNER.
        material (Material, optional): The material of the box. Defaults to AIR.
        thickness (float, optional): The thickness of the PML Layer. Defaults to 0.1.
        Nlayers (int, optional): The number of geometrical PML layers. Defaults to 1.
        N_mesh_layers (int, optional): The number of mesh layers. Sets the discretization size accordingly. Defaults to 5
        exponent (float, optional): The PML gradient growth function. Defaults to 1.5.
        deltamax (float, optional): A PML matching coefficient. Defaults to 8.0.
        sides (str, optional): A string of pml sides as characters ([T]op, [B]ottom, [L]eft, [R]ight, [F]ront, b[A]ck)
        top (bool, optional): Add a top PML layer. Defaults to True.
        bottom (bool, optional): Add a bottom PML layer. Defaults to False.
        left (bool, optional): Add a left PML layer. Defaults to False.
        right (bool, optional): Add a right PML layer. Defaults to False.
        front (bool, optional): Add a front PML layer. Defaults to False.
        back (bool, optional): Add a back PML layer. Defaults to False.

    Returns:
        list[GeoVolume]: A list of objects [main box, *pml boxes]
    """
    
    sides = sides.lower()

    top    = "t" in sides or top
    bottom = "b" in sides or bottom
    left   = "l" in sides or left
    right  = "r" in sides or right
    front  = "f" in sides or front
    back   = "a" in sides or back
    

    px, py, pz = position
    if alignment == Alignment.CORNER:
        px = px + width / 2
        py = py + depth / 2
        pz = pz + height / 2

    position = (px, py, pz)
    main_box = Box(width, depth, height, position, alignment=Alignment.CENTER)
    main_box.properties += material

    main_box._unset_constraints = True
    other_boxes = []

    addpml = partial(_add_pml_layer, center=(px, py, pz), dims=(width, depth, height),
                     thickness=thickness, Nlayers=Nlayers, N_mesh_layers=N_mesh_layers,
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
    
    return [main_box] + other_boxes