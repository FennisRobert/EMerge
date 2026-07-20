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

__version__ = "2.1.1"

NTHREADS = "1"
os.environ["EMERGE_STD_LOGLEVEL"] = os.getenv("EMERGE_STD_LOGLEVEL", default="INFO")
os.environ["EMERGE_FILE_LOGLEVEL"] = os.getenv("EMERGE_FILE_LOGLEVEL", default="DEBUG")
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", default="8")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", default="4")
os.environ["OPENBLAS_NUM_THREADS"] = os.getenv("OPENBLAS_NUM_THREADS", default=NTHREADS)
os.environ["VECLIB_NUM_THREADS"] = NTHREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NTHREADS
os.environ["NUMEXPR_NUM_THREADS"] = NTHREADS
os.environ["NUMBA_NUM_THREADS"] = os.getenv("NUMBA_NUM_THREADS", default="4")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")


############################################################
#                      IMPORT MODULES                     #
############################################################

from ._emerge.logsettings import _GlobalHandler.active().logcontroller
from loguru import logger

_GlobalHandler.active().logcontroller.set_default()
logger.debug('Importing modules')
_GlobalHandler.active().logcontroller._set_log_buffer()

import gmsh
from ._emerge.simmodel import Simulation, SimulationBeta
#from ._emerge.material import Material, FreqCoordDependent, FreqDependent, CoordDependent
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