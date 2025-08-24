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

from .pcb import PCB
from .pmlbox import pmlbox
from .horn import Horn
from .shapes import Cylinder, CoaxCylinder, Box, XYPlate, HalfSphere, Sphere, Plate, OldBox, Alignment, Cone
from .operations import subtract, add, embed, remove, rotate, mirror, change_coordinate_system, translate, intersect, unite, expand_surface
from .polybased import XYPolygon, GeoPrism, Disc, Curve
from .step import STEPItems