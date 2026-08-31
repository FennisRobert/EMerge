# Last Cleanup: 2025-01-01
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

__version__ = "3.0.0a14"

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
from loguru import logger
logger.info(f'EMerge v{__version__}')
logger.debug('Importing modules')

import gmsh
from ._emerge.simmodel import Simulation
from ._emerge.solver import SolverBicgstab, SolverGMRES, SolveRoutine, ReverseCuthillMckee, Sorter, SolverPardiso, SolverUMFPACK, SolverSuperLU, EMSolver
from ._emerge.cs import CoordinateSystem, CS, GCS, Plane, Axis, XAX, YAX, ZAX, XYPLANE, XZPLANE, YZPLANE, YXPLANE, ZXPLANE, ZYPLANE, cs
from ._emerge.coord import Line
from ._emerge.geo.pcb import PCB, PCBLayer, PCBNew
from ._emerge.geo.pmlbox import pmlbox
from ._emerge.geo.horn import Horn
from ._emerge.geo.shapes import Cylinder, CoaxCylinder, Box, XYPlate, HalfSphere, Sphere, Plate, OldBox, Alignment, Cone
from ._emerge.geo.operations import subtract, add, embed, remove, rotate, mirror, change_coordinate_system, translate, intersect, unite, expand_surface, stretch, extrude, stick, bounding_box
from ._emerge.geo.polybased import XYPolygon, GeoPrism, Disc, Curve
from ._emerge.geo.step import STEPItems
from ._emerge.geo.open_region import open_region, open_pml_region
from ._emerge.selection import Selection, FaceSelection, DomainSelection, EdgeSelection
from ._emerge.geometry import select
from ._emerge.mth.common_functions import norm, coax_rout, coax_rin, dot, cross
from ._emerge.periodic import RectCell, HexCell
from ._emerge.mesher import Algorithm2D, Algorithm3D
from ._emerge.howto import _HowtoClass
from ._emerge.emerge_update import update_emerge
from ._emerge.cleanup import cleanup
from .auxilliary.touchstone import TouchstoneData
from emsutil import isola, rogers, const, lib
from emsutil.material import Material, MatProperty, FreqDependent, CoordDependent, FreqCoordDependent
from emsutil.plot.plot2d import plot, plot_ff, plot_ff_polar, plot_sp, plot_vswr, smith 
from emsutil import EMergeTheme
from emsutil import themes

howto = _HowtoClass()

logger.debug('Importing complete!')

from ._emerge.install_check import run_installation_checks

run_installation_checks()


############################################################
#                      GLOBAL HANDLER                     #
############################################################

from ._emerge._global import _GlobalHandler, cleanup
from ._emerge.geo.pcb import _PCBManager
from ._emerge.simstate import _SimStateManager
from ._emerge.selection import Selector
from ._emerge.logsettings import LogController, DebugCollector

GLOBALHANDLER = _GlobalHandler()
GLOBALHANDLER.geomanager = _GeometryManager()
GLOBALHANDLER.generator = _KeyGenerator()
GLOBALHANDLER.pcbmanager = _PCBManager()
GLOBALHANDLER.simstates = _SimStateManager()
GLOBALHANDLER.selector = Selector()
GLOBALHANDLER.logcontroller = LogController()
GLOBALHANDLER.debugcollector = DebugCollector()
# Install global states

############################################################
#                         CONSTANTS                        #
############################################################

CENTER = Alignment.CENTER
"""Center alignment enum."""

CORNER = Alignment.CORNER
"""Corner alignment enum."""

EISO = lib.EISO
"""Divide the far-field E-field by this to obtain isotropic gain."""

EOMNI = lib.EOMNI
"""Divide the far-field E-field by this to obtain omnidirectional gain."""

PI = lib.PI
"""π (3.141592653589793...)."""

mm = 0.001
"""Millimeter (m)."""

kHz = 1_000.0
"""Kilohertz (Hz)."""

MHz = 1_000_000.0
"""Megahertz (Hz)."""

GHz = 1_000_000_000.0
"""Gigahertz (Hz)."""

THz = 1_000_000_000_000.0
"""Terahertz (Hz)."""

inch = 0.0254
"""Inch (25.4 mm)."""

mil = 0.0000254
"""Mil (one thousandth of an inch)."""