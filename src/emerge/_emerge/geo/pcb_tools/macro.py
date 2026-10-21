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

# Last Cleanup: 2025-01-01
from typing import Any
from dataclasses import dataclass
import math
import re

def rotation_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Returns the signed angle in degrees you must rotate vector a
    to align with vector b. Positive = CCW, Negative = CW.
    
    a, b: (vx, vy) tuples in the plane.

    Args:
        a (tuple[float, float]): The (vx, vy) start vector
        b (tuple[float, float]): The (vx, vy) target vector

    Returns:
        float: The signed rotation angle in degrees
    """
    ax, ay = a
    bx, by = b

    # compute 2D cross product (scalar) and dot product
    cross = ax * by - ay * bx
    dot   = ax * bx + ay * by

    # atan2(cross, dot) gives angle between -π and π
    angle_rad = math.atan2(cross, dot)
    angle_deg = math.degrees(angle_rad)
    return -angle_deg

@dataclass
class Instruction:
    """A single StripPath method call produced by parse_macro.

    Args:
        instr (str): The StripPath method name
        args (tuple[int | float, ...]): The positional arguments
        kwargs (dict[str,float] | None, optional): The keyword arguments. Defaults to None.
    """
    instr: str
    args: tuple[int | float, ...]
    kwargs: dict[str,float] | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

symbols = ['=','>','v','^','<','@','/','\\','T']
char_class = ''.join(re.escape(c) for c in symbols)

pattern = re.compile(rf'([{char_class}])([\d\,\.\-]+)')

def parse_macro(pathstring: str, width: int | float, direction: tuple[float, float]) -> list[Instruction]:
    """Parse an EMerge macro command string into StripPath instructions (see StripPath.macro for the language).

    Args:
        pathstring (str): The macro command string
        width (int | float): The start trace width
        direction (tuple[float, float]): The start direction

    Returns:
        list[Instruction]: The StripPath method calls to execute
    """
    instructions = pattern.findall(pathstring.replace(' ',''))
    
    oi = []
    for com, val in instructions:
        if ',' in val:
            val_list: list[float] = [float(x) for x in val.split(',')]
            ival, width = val_list[0], val_list[1]
        else:
            ival = float(val)
        if com == '=':
            oi.append(Instruction('straight',(ival,),{'width': width})) #type: ignore
        elif com == '>':
            oi.append(Instruction('turn',(rotation_angle(direction, (1., 0.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (1. ,0. )
        elif com == '<':
            oi.append(Instruction('turn',(rotation_angle(direction, (-1., 0.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (-1.,0.)
        elif com == 'v':
            oi.append(Instruction('turn',(rotation_angle(direction, (0.,-1.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (0.,-1.)
        elif com == '^':
            oi.append(Instruction('turn',(rotation_angle(direction, (0.,1.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (0.,1.)
        elif com == '\\':
            oi.append(Instruction('turn',(90,) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (direction[1],-direction[0])
        elif com == '/':
            oi.append(Instruction('turn',(-90,) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (-direction[1],direction[0])
        elif com == 'T':
            oi.append(Instruction('taper',(ival,),{'width': width}))
        elif com == '@':
            oi.append(Instruction('turn',(ival,),{'width': width}))
    return oi