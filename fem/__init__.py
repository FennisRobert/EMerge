from loguru import logger
import sys

from .simmodel import Simulation3D
from . import material
from . import physics
from . import bc
from .solver import SolverBicgstab, SolverGMRES, SolveRoutine, ReverseCuthillMckee, SolverAMG, Sorter, SolverPardiso, SolverSP
from .cs import CoordinateSystem, Plane, Axis, XAX, YAX, ZAX, XYPLANE, XZPLANE, YZPLANE, YXPLANE, ZXPLANE, ZYPLANE
from .coord import Line
from . import plot
from . import modeling
from .selection import Selection, FaceSelection, DomainSelection, EdgeSelection
from .mth.common_functions import norm
from .logsettings import logger_format

logger.remove()
logger.add(sys.stderr, format=logger_format)