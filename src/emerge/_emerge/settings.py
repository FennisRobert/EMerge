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
from typing import Literal


class Settings:
    def __init__(self):
        self.mw_2dbc: bool = True
        # Automatically assign 2D boundary conditions based on material properties.

        self.mw_2dbc_lim: float = 10.0
        # Bulk conductivity limit (S/m) beyond which a surface material is assigned
        # a SurfaceImpedance boundary condition.

        self.mw_2dbc_peclim: float = 1e8
        # Bulk conductivity limit (S/m) beyond which a conductor is assigned PEC
        # instead of a SurfaceImpedance boundary condition.

        self.mw_3d_peclim: float = 1e8
        # Bulk conductivity limit (S/m) beyond which 3D conductors are considered PEC.

        self.mw_3d_surfimplim: float = 1e4
        # Bulk conductivity limit beyond which a volume is no longer assembled but treated with a SurfaceImpedance boundary condition

        self.mw_cap_sp_single: bool = True
        # Cap single S-parameters to a maximum magnitude of 1.0.

        self.mw_cap_sp_col: bool = True
        # Power-normalize S-parameter columns to 1.0.

        self.mw_recip_sp: bool = False
        # Explicitly enforce reciprocity on S-parameters.

        self.sim_symmetry_limit: float = 0.02
        # Threshold for the symmetry handling of direct solvers. This ratio defines the maximum allowed
        # ratio between the highest Sparse matrix entry and the difference between any two items ij and ji:
        # max(abs(Mij - Mji))/max(abs(M)) < ratio

        self.size_check: bool = True
        # Perform total volume check (≈100,000 tetrahedra limit).
        # ~100k tetrahedra ≈ 700k DOF. Prevents oversized simulations.

        self.auto_save: bool = False
        # Automatically save if simulation aborts unexpectedly.

        self.save_after_sim: bool = True
        # Save simulation only if it completes successfully.

        self.save_method: Literal["joblib", "msgpack"] = "joblib"
        # Serialization method used for saving simulations.

        self.check_ram: bool = False
        # If a RAM memory check should be done and halt the solver.

        self.safe_mode: bool = False
        # Executes extra diagnosis checks to ensure a proper matrix problem

        self.qtem_limit: float = 0.1


DEFAULT_SETTINGS = Settings()
