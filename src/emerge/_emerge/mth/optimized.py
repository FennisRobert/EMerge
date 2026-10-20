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
import numpy as np
from .keast_rules import _KEAST_PTS
from .strang_rules import _STRANG_PTS

############################################################
#                         FUNCTIONS                        #
############################################################


def gaus_quad_tri(p: int) -> np.ndarray:
    """
    Returns the duvanant quadrature triangle sample points W, L1, L2, L3, coordinates for a given order p.

    Parameters
    ----------
    p : int
        The order of the quadrature rule.
    Returns
    -------
    pts : np.ndarray
        The sample points W, L1, L2, L3.
    -------

    P = dunavant_points(p)
    P[0,:] = Weights
    P[1,:] = L1 values
    P[2,:] = L2 values
    P[3,:] = L3 values
    """

    return _STRANG_PTS[p]


def gaus_quad_tet(p: int) -> np.ndarray:
    """
    Returns the duvanant quadrature tetrahedron sample points W, L1, L2, L3, L4, coordinates for a given order p.

    Parameters
    ----------
    p : int
        The order of the quadrature rule.
    Returns
    -------
    pts : np.ndarray
        The sample points W, L1, L2, L3.
    -------

    P = dunavant_points(p)
    P[0,:] = Weights
    P[1,:] = L1 values
    P[2,:] = L2 values
    P[3,:] = L3 values
    """

    return _KEAST_PTS[p]
