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
from .pcb import PCB, PCBLayer, PCBNew
from .pmlbox import pmlbox
from .horn import Horn
from .shapes import Cylinder, CoaxCylinder, Box, XYPlate, HalfSphere, Sphere, Plate, OldBox, Alignment, Cone
from .operations import duplicate, subtract, add, embed, remove, rotate, mirror, change_coordinate_system, translate, intersect, unite, expand_surface, stretch, extrude, stick, bounding_box, connect_faces, thick_wall
from .polybased import XYPolygon, GeoPrism, Disc, Curve
from .step import STEPItems
from .open_region import open_region, open_pml_region