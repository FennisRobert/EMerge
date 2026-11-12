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

# [Written by Claude]
"""The solvers of EMerge.

Select one with model.set_solver(em.solvers.Pardiso). Every solver is listed here; the ones that need a library
that is not installed raise an error when selected. available() lists the solvers that can be used on this system.

Direct solvers: SuperLU, UMFPACK, Pardiso, MUMPS, AASDS, CuDSS, CHOLMOD, RSLab, Sparta.
Iterative solvers: BiCGSTAB, CG, Corvid.
Eigenvalue solvers: LAPACK, ARPACK.
"""
from ._emerge.solvers import (
    SolverSuperLU as SuperLU,
    SolverUMFPACK as UMFPACK,
    SolverPardiso as Pardiso,
    SolverMUMPS as MUMPS,
    SolverAASDS as AASDS,
    SolverCuDSS as CuDSS,
    SolverCHOLMOD as CHOLMOD,
    SolverRSLAB as RSLab,
    SolverSparta as Sparta,
    SolverBicgstab as BiCGSTAB,
    SolverCG as CG,
    SolverCorvid as Corvid,
    SolverLAPACK as LAPACK,
    SolverARPACK as ARPACK,
    available,
)

__all__ = [
    "SuperLU", "UMFPACK", "Pardiso", "MUMPS", "AASDS", "CuDSS", "CHOLMOD", "RSLab", "Sparta",
    "BiCGSTAB", "CG", "Corvid", "LAPACK", "ARPACK", "available",
]
