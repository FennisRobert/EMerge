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
"""
THESE FUNCTIONS ARE MEANT TO QUICKLY COUPLE COORDINATES ON TWO PERIODIC
BOUNDARIES IN THE SMALLEST AMOUNT OF TIME WITHOUT RISKING FLOAT ROUNDING ERRORS.

THE MAPPING IS DONE BY BY SIMPLY FINDING THE CLOSEST COORDINATE AFTER PROJECTION
BY SEARCHING A LIST SORTED DICTIONARY WISE ON 3D COORDINATES.

ALL PAIRED NODES ARE NO LONGER VISITED
"""

import numpy as np
from emr.mth import link_coords


############################################################
#                   MAIN PYTHON INTERFACE                  #
############################################################

def pair_coordinates(coords: np.ndarray, ids1: np.ndarray, ids2: np.ndarray, disp: np.ndarray, dsmax: float) -> dict[int, int]:
    """ This function finds the mapping between a total coordinate set and two lits of indices.

    The indices correspond to two faces that are identical but displaced (mesh centroids of periodic boundaries).

    Args:
        coords (np.ndarray): A total set of coordinates
        ids1 (list[int]): The indices of the source set
        ids2 (list[int]): The indices of the to-be-matched set
        disp (np.ndarray): The displacement vector of shape (3,)
        dsmax (float): The maximum allowed displacement in matchiing

    Returns:
        dict[int, int]: An int,int mapping of the indices.
    """
    ids1_c_sorted = sorted(ids1, key= lambda x: tuple(coords[:,x]))
    ids2_c_sorted = sorted(ids2, key= lambda x: tuple(coords[:,x]-disp))

    mapping = link_coords(coords, np.array(ids1_c_sorted), np.array(ids2_c_sorted), disp, dsmax)
    
    mapping = {i: j for i,j in zip(mapping[0,:], mapping[1,:])}
    
    return mapping