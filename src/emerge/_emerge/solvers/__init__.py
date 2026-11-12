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

"""The linear and eigenvalue solvers of EMerge.

Each solver lives in its own module, together with the interface to its library. A solver class says whether its
library is installed (`available`) and what it needs otherwise (`requires`).
"""
from __future__ import annotations
from enum import Enum, EnumMeta
from loguru import logger

from .solver_base import (
    MatrixType, classify_matrix, SolveReport, _pfx, is_numerically_complex_symmetric, complex_to_real_block,
    real_to_complex_block, SolverType, SimulationError, Sorter, Preconditioner, Solver, EigSolver,
)
from .preconditioners import ReverseCuthillMckee, ILUPrecon, JacobiPrecon, ICCPrecon
from .iterative import SolverCG, SolverBicgstab, SolverGCROTMK, SolverGMRES
from .cholmod import SolverCHOLMOD
from .superlu import SolverSuperLU
from .umfpack import SolverUMFPACK
from .rslab import SolverRSLAB
from .sparta import SolverSparta
from .corvid import SolverCorvid
from .mumps import SolverMUMPS
from .aasds import SolverAASDS
from .pardiso import SolverPardiso
from .eigen import SolverLAPACK, SolverARPACK
from .boundary_modes import SolverARPACK_BMA

# The cuDSS interface needs the CUDA stack when it is imported. Without it SolverCuDSS is a placeholder, so the
# name always exists and selecting it gives a clear error.
try:
    from .cudss import SolverCuDSS

    SolverCuDSS.available = True
except Exception as e:
    if not isinstance(e, ModuleNotFoundError):
        # Any failure in the optional GPU stack (e.g. an API change in nvmath-python raising an
        # AttributeError) must not break `import emerge` for CPU-only use.
        logger.warning(f"The cuDSS GPU solver is unavailable: {type(e).__name__}: {e}")
        logger.debug("cuDSS import error details:", exc_info=e)
    else:
        logger.trace("CuDSS Interface not found. (not harmful)")

    class SolverCuDSS(Solver):  # type: ignore[no-redef]
        """Implements the NVIDIA cuDSS GPU sparse direct solver.

        Requires an NVIDIA GPU and pip install emerge[cudss].
        """

        name = "CUDSS"
        available = False
        requires = "an NVIDIA GPU and pip install emerge[cudss]"


# [Written by Claude] The solvers a SolveRoutine creates (when available), in order of preference for the fallback.
ALL_SOLVERS: tuple[type[Solver] | type[EigSolver], ...] = (
    SolverSuperLU, SolverUMFPACK, SolverPardiso, SolverMUMPS, SolverAASDS, SolverCuDSS, SolverCHOLMOD, SolverRSLAB,
    SolverSparta, SolverCorvid, SolverBicgstab, SolverCG, SolverLAPACK, SolverARPACK, SolverARPACK_BMA,
)


# The solvers a user can select, by their name in em.solvers
PUBLIC_SOLVERS: dict[str, type[Solver] | type[EigSolver]] = {
    "SuperLU": SolverSuperLU, "UMFPACK": SolverUMFPACK, "Pardiso": SolverPardiso, "MUMPS": SolverMUMPS,
    "AASDS": SolverAASDS, "CuDSS": SolverCuDSS, "CHOLMOD": SolverCHOLMOD, "RSLab": SolverRSLAB, "Sparta": SolverSparta,
    "BiCGSTAB": SolverBicgstab, "CG": SolverCG, "Corvid": SolverCorvid, "LAPACK": SolverLAPACK, "ARPACK": SolverARPACK,
}


def available() -> list[type[Solver] | type[EigSolver]]:
    """The solvers of em.solvers that can be used on this system.

    Returns:
        list[type[Solver] | type[EigSolver]]: The solver classes whose library is installed
    """
    return [cls for cls in PUBLIC_SOLVERS.values() if cls.available]


def unavailable_message(cls: type[Solver] | type[EigSolver]) -> str:
    """The error message for a solver that is not installed on this system."""
    names = ", ".join(f"em.solvers.{name}" for name, c in PUBLIC_SOLVERS.items() if c.available)
    return f"The {cls.name} solver is not available on this system. It requires {cls.requires}. Available: {names}."


############################################################
#                 REMOVED: THE EMSolver ENUM               #
############################################################

# [Written by Claude] Removed in 3.0: a solver is selected with its class, e.g. model.set_solver(em.solvers.Pardiso).
# The enum stays for a few versions with its old members, so old scripts still autocomplete, but accessing a member
# (em.EMSolver.PARDISO) raises an error that names the replacement.
_EMSOLVER_REPLACEMENTS = {
    "SUPERLU": "SuperLU", "UMFPACK": "UMFPACK", "PARDISO": "Pardiso", "LAPACK": "LAPACK", "ARPACK": "ARPACK",
    "CUDSS": "CuDSS", "MUMPS": "MUMPS", "AASDS": "AASDS", "BICGSTAB": "BiCGSTAB", "CG": "CG", "CHOLMOD": "CHOLMOD",
    "RSLAB": "RSLab", "SPARTA": "Sparta", "CORVID": "Corvid",
}


def _emsolver_removed(name: str) -> RuntimeError:
    new = _EMSOLVER_REPLACEMENTS.get(name)
    hint = f"em.solvers.{new}" if new else "a class from em.solvers"
    return RuntimeError(
        f"em.EMSolver was removed in EMerge 3.0. Replace em.EMSolver.{name} with {hint}, "
        f"for example model.set_solver({hint}). em.solvers.available() lists the solvers installed on this system."
    )


class _RemovedEnumMeta(EnumMeta):
    def __getattribute__(cls, name: str):
        if not name.startswith("_") and name in type.__getattribute__(cls, "_member_map_"):
            raise _emsolver_removed(name)
        return super().__getattribute__(name)


class EMSolver(Enum, metaclass=_RemovedEnumMeta):
    """Removed in EMerge 3.0. Select a solver with a class from em.solvers, e.g. model.set_solver(em.solvers.Pardiso).

    em.solvers.available() lists the solvers that are installed on this system.
    """

    TEST = 0
    SUPERLU = 1
    UMFPACK = 2
    PARDISO = 3
    LAPACK = 4
    ARPACK = 5
    SMART_ARPACK_BMA = 6
    CUDSS = 7
    MUMPS = 8
    AASDS = 9
    BICGSTAB = 10
    CG = 11
    CHOLMOD = 12
    RSLAB = 13
    SPARTA = 14
    CORVID = 15


def reject_emsolver(solver: object) -> None:
    """Raise the removal error for a member of the removed EMSolver enum (e.g. from EMSolver['PARDISO'])."""
    if isinstance(solver, EMSolver):
        raise _emsolver_removed(solver._name_)


__all__ = [
    "MatrixType", "classify_matrix", "SolveReport", "_pfx", "is_numerically_complex_symmetric", "complex_to_real_block",
    "real_to_complex_block", "SolverType", "SimulationError", "Sorter", "Preconditioner", "Solver", "EigSolver",
    "ReverseCuthillMckee", "ILUPrecon", "JacobiPrecon", "ICCPrecon",
    "SolverCG", "SolverBicgstab", "SolverGCROTMK", "SolverGMRES", "SolverCHOLMOD", "SolverSuperLU", "SolverUMFPACK",
    "SolverRSLAB", "SolverSparta", "SolverCorvid", "SolverMUMPS", "SolverAASDS", "SolverPardiso", "SolverCuDSS",
    "SolverLAPACK", "SolverARPACK", "SolverARPACK_BMA", "ALL_SOLVERS", "PUBLIC_SOLVERS", "available", "unavailable_message", "EMSolver", "reject_emsolver",
]
