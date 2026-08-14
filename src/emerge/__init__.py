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
############################################################
#                    WARNING SUPPRESSION                   #
############################################################

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type swigvarlink.*"
)

############################################################
#               HANDLE ENVIRONMENT VARIABLES              #
############################################################

import os

__version__ = "2.8.4"

NTHREADS = "1"
os.environ.setdefault("EMERGE_STD_LOGLEVEL", "INFO")
os.environ.setdefault("EMERGE_FILE_LOGLEVEL", "DEBUG")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", NTHREADS)
os.environ.setdefault("VECLIB_NUM_THREADS", NTHREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", NTHREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", NTHREADS)
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

############################################################
#                      IMPORT MODULES                     #
############################################################

from ._emerge.logsettings import LOG_CONTROLLER
from loguru import logger

LOG_CONTROLLER.set_default()
logger.info(f'EMerge v{__version__}')
logger.debug('Importing modules')
LOG_CONTROLLER._set_log_buffer()

import gmsh
from ._emerge.simmodel import Simulation, SimulationBeta
from ._emerge import bc
from ._emerge.solver import SolverBicgstab, SolverGMRES, SolveRoutine, ReverseCuthillMckee, Sorter, SolverPardiso, SolverUMFPACK, SolverSuperLU, EMSolver
from ._emerge.cs import CoordinateSystem, CS, GCS, Plane, Axis, XAX, YAX, ZAX, XYPLANE, XZPLANE, YZPLANE, YXPLANE, ZXPLANE, ZYPLANE, cs, Anchor
from ._emerge.coord import Line
from ._emerge import geo
from ._emerge.selection import Selection, FaceSelection, DomainSelection, EdgeSelection
from ._emerge.geometry import select
from ._emerge.mth.common_functions import norm, coax_rout, coax_rin, dot, cross, lumped_element_material
from ._emerge.periodic import RectCell, HexCell
from ._emerge.mesher import Algorithm2D, Algorithm3D
from ._emerge.howto import _HowtoClass
from ._emerge.emerge_update import update_emerge
from ._emerge.cleanup import cleanup
from . import integrals as intf
from .auxilliary.touchstone import TouchstoneData
from emsutil import isola, rogers, const, lib
from emsutil.material import Material, MatProperty, FreqDependent, CoordDependent, FreqCoordDependent
import emsutil.plot as plot
from emsutil import EMergeTheme
from emsutil import themes
from emsutil.lib import C0, MU0, EPS0, Z0
from ._emerge.elements.dofsets import ElementSpace, DoFSet

howto = _HowtoClass()

from ._emerge.install_check import run_installation_checks

run_installation_checks()

############################################################
#                         CONSTANTS                        #
############################################################

CENTER = geo.Alignment.CENTER
CORNER = geo.Alignment.CORNER
EISO = lib.EISO
EOMNI = lib.EOMNI
