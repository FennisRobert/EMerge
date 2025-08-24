"""A Python based FEM solver.
Copyright (C) 2025 Robert Fennis

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
import os

__version__ = "0.6.7"

############################################################
#               HANDLE ENVIRONMENT VARIABLES              #
############################################################

NTHREADS = "1"
os.environ["EMERGE_STD_LOGLEVEL"] = os.getenv("EMERGE_STD_LOGLEVEL", default="INFO")
os.environ["EMERGE_FILE_LOGLEVEL"] = os.getenv("EMERGE_FILE_LOGLEVEL", default="DEBUG")
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", default="4")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", default="4")
os.environ["OPENBLAS_NUM_THREADS"] = NTHREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NTHREADS
os.environ["NUMEXPR_NUM_THREADS"] = NTHREADS


############################################################
#                      IMPORT MODULES                     #
############################################################
from ._emerge import _cache_check
from ._emerge.logsettings import LOG_CONTROLLER
from loguru import logger

LOG_CONTROLLER.set_default()
logger.debug('Importing modules')

from ._emerge.simmodel import Simulation
from ._emerge.material import Material
from ._emerge import bc
from ._emerge.solver import SolverBicgstab, SolverGMRES, SolveRoutine, ReverseCuthillMckee, Sorter, SolverPardiso, SolverUMFPACK, SolverSuperLU, EMSolver
from ._emerge.cs import CoordinateSystem, CS, GCS, Plane, Axis, XAX, YAX, ZAX, XYPLANE, XZPLANE, YZPLANE, YXPLANE, ZXPLANE, ZYPLANE
from ._emerge.coord import Line
from ._emerge import geo
from ._emerge.selection import Selection, FaceSelection, DomainSelection, EdgeSelection
from ._emerge.geometry import select
from ._emerge.mth.common_functions import norm, coax_rout, coax_rin
from ._emerge.physics.microwave.sc import stratton_chu
from ._emerge.periodic import RectCell, HexCell
from ._emerge.mesher import Algorithm2D, Algorithm3D
from . import lib
from ._emerge.howto import _HowtoClass

howto = _HowtoClass()

logger.debug('Importing complete!')