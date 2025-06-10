"""A Python based FEM solver.
Copyright (C) 2025  name of Robert Fennis

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see
<https://www.gnu.org/licenses/>.

"""

from loguru import logger
from .logsettings import logger_format
import sys

logger.remove()
logger.add(sys.stderr, format=logger_format)

logger.warning('Note that the first time running this library can take a couple of seconds.' \
'Numba has to compile C-compiled functions of a bunch of scripts. These compilations will be cached locally' \
'so you dont have to wait this long on subsequent runs. However, any time you make a single character change' \
'to a script containing numba compiled functions means a complete recompilation of all functions in that script.')

from .simmodel import Simulation3D
from .material import Material, FR4, AIR, VACUUM, COPPER
from . import physics
from . import bc
from .solver import SolverBicgstab, SolverGMRES, SolveRoutine, ReverseCuthillMckee, Sorter, SolverPardiso, SolverSP
from .cs import CoordinateSystem, Plane, Axis, XAX, YAX, ZAX, XYPLANE, XZPLANE, YZPLANE, YXPLANE, ZXPLANE, ZYPLANE
from .coord import Line
from . import plot
from . import modeling
from .selection import Selection, FaceSelection, DomainSelection, EdgeSelection
from .mth.common_functions import norm
