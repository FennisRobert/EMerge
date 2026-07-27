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
from .pmlbox import pmlbox
from .shapes import Box, GeoVolume
from .operations import bounding_box
from emsutil import Material, AIR
from numbers import Number
import gmsh

def open_region(xmargin: float | tuple[float, float],
                ymargin: float | tuple[float, float],
                zmargin: float | tuple[float, float],
                material: Material = AIR) -> Box:
    """Creates a simple open region air box with the given margins.

    Args:
        xmargin (float | tuple[float, float]): The X-margin as a single number of the -x and +x margin.
        ymargin (float | tuple[float, float]): The Y-margin as a single number of the -y and +y margin.
        zmargin (float | tuple[float, float]): The Z-margin as a single number of the -z and +z margin.
        material (Material): The objects material. Defaults to AIR.
    
    Returns:
        Box: The airbox geoemtry object
    """
    gmsh.model.occ.synchronize()
    if isinstance(xmargin, Number):
        xmargin = (xmargin, xmargin)
    if isinstance(ymargin, Number):
        ymargin  = (ymargin, ymargin)
    if isinstance(zmargin, Number):
        zmargin = (zmargin, zmargin)
        
    xm1, xm2 = xmargin
    ym1, ym2 = ymargin
    zm1, zm2 = zmargin

    x1, x2, y1, y2, z1, z2 = bounding_box()
    
    W = x2-x1 + xm1 + xm2
    D = y2-y1 + ym1 + ym2
    H = z2-z1 + zm1 + zm2

    return Box(W,D,H, position=(x1-xm1, y1-ym1, z1-zm1), name='OpenRegion').set_material(material).background()


def open_pml_region(xmargin: float | tuple[float, float],
                    ymargin: float | tuple[float, float],
                    zmargin: float | tuple[float, float],
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
    
    """Creates a simple open region air box with PML layers with the given margins.

    Args:
        xmargin (float | tuple[float, float]): The X-margin as a single number of the -x and +x margin.
        ymargin (float | tuple[float, float]): The Y-margin as a single number of the -y and +y margin.
        zmargin (float | tuple[float, float]): The Z-margin as a single number of the -z and +z margin.
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
        _type_: _description_
    """
    if isinstance(xmargin, Number):
        xmargin = (xmargin, xmargin)
    if isinstance(ymargin, Number):
        ymargin  = (ymargin, ymargin)
    if isinstance(zmargin, Number):
        zmargin = (zmargin, zmargin)
        
    xm1, xm2 = xmargin
    ym1, ym2 = ymargin
    zm1, zm2 = zmargin

    x1, x2, y1, y2, z1, z2 = bounding_box()
    
    W = x2-x1 + xm1+xm2
    D = y2-y1 + ym1+ym2
    H = z2-z1 + zm1+zm2

    return pmlbox(W,D,H, 
                  position=(x1-xm1, y1-ym1, z1-zm1), 
                  material = material,
                  thickness = thickness,
                  Nlayers = Nlayers,
                  N_mesh_layers = N_mesh_layers,
                  exponent = exponent,
                  deltamax = deltamax,
                  sides = sides,
                  top = top,
                  bottom = bottom,
                  left = left,
                  right = right,
                  front = front,
                  back = back)

