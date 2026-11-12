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

"""Base classes of the linear and eigenvalue solvers, the solve report and the matrix types."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, ClassVar
import numpy as np
from scipy import sparse  # type: ignore
from scipy.sparse import csc_matrix  # type: ignore
from loguru import logger
from ..file import Saveable


############################################################
#                       MATRIX TYPES                      #
############################################################


class MatrixType(Enum):
    """Classification of the global system matrix for solver selection.

    The matrix type determines which solvers and preconditioners are valid:
      - SPD:                CHOLMOD (Cholesky)
      - SYMMETRIC:          LU / BiCGSTAB
      - HERMITIAN:          not used yet
      - HPD:                not used yet
      - COMPLEX_SYMMETRIC:  LU
      - UNSYMMETRIC:        LU
    """

    SPD = 0
    SYMMETRIC = 1
    HPD = 2
    HERMITIAN = 3
    COMPLEX_SYMMETRIC = 4
    UNSYMMETRIC = 5
    UNKNOWN = 6

    @property
    def symmetric(self) -> bool:
        if self.value in (0, 1, 4):
            return True
        return False


def classify_matrix(A: csc_matrix, tol: float = 1e-10) -> MatrixType:
    """Classifies the matrix type of matrix A"""

    # If A is non-square, its definitely not symmetric
    n, m = A.shape
    if n != m:
        return MatrixType.UNSYMMETRIC

    is_complex = np.iscomplexobj(A.data)

    if not is_complex:
        # Real matrix — check symmetry: A = A^T
        diff = A - A.T
        if diff.nnz == 0 or abs(diff).max() < tol:
            # Symmetric — check positive definiteness via diagonal
            diag = A.diagonal()
            if np.all(diag > 0):
                return MatrixType.SPD
            else:
                return MatrixType.SYMMETRIC
        else:
            return MatrixType.UNSYMMETRIC
    else:
        # Complex matrix — check complex symmetry: A = A^T (not conjugate)
        diff_sym = A - A.T
        is_sym = diff_sym.nnz == 0 or abs(diff_sym).max() < tol

        # Check Hermitian: A = A^H
        diff_herm = A - A.conj().T
        is_herm = diff_herm.nnz == 0 or abs(diff_herm).max() < tol

        if is_herm:
            diag = A.diagonal().real
            if np.all(diag > 0) and np.all(np.abs(A.diagonal().imag) < tol):
                return MatrixType.HPD
            else:
                return MatrixType.HERMITIAN
        elif is_sym:
            return MatrixType.COMPLEX_SYMMETRIC
        else:
            return MatrixType.UNSYMMETRIC


############################################################
#                       SOLVE REPORT                       #
############################################################


@dataclass
class SolveReport(Saveable):
    simtime: float = -1.0
    jobid: int = -1
    ndof: int = -1
    nnz: int = -1
    ndof_solve: int = -1
    nnz_solve: int = -1
    exit_code: int = 0
    solver: str = "None"
    sorter: str = "None"
    precon: str = "None"
    aux: dict[str, str] = field(default_factory=dict)
    worker_name: str = "Unknown Worker"

    def add(self, **kwargs: str):
        for key, value in kwargs.items():
            self.aux[key] = str(value)

    @property
    def mdof(self) -> float:
        """Performance metric in MDoF per second"""
        return (self.ndof**2) / ((self.simtime + 1e-6) * 1e6)

    def logprint(self, print_cal: Callable | None = None):
        if print_cal is None:
            print_cal = print

        def fmt(key, val):
            return f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}"

        parts = []
        parts.append(fmt("Solver", self.solver))
        parts.append(fmt("Sorter", self.sorter))
        parts.append(fmt("Precon", self.precon))
        parts.append(fmt("JobID", self.jobid))
        parts.append(fmt("SimTime[s]", self.simtime))
        parts.append(fmt("DOFsTot", self.ndof))
        parts.append(fmt("NNZTot", self.nnz))
        parts.append(fmt("DOFsSolve", self.ndof_solve))
        parts.append(fmt("NNZSolve", self.nnz_solve))
        parts.append(fmt("Exit", self.exit_code))
        parts.append(fmt("Worker", self.worker_name))

        if self.aux:
            for k, v in self.aux.items():
                parts.append(fmt(str(k), v))

        # Group into multiple lines (6 items per line for readability)
        print_cal(f"FEM Report [JobID={self.jobid}]")
        for i in range(0, len(parts), 6):
            print_cal("  " + ", ".join(parts[i : i + 6]))

    def pretty_print(self, print_cal: Callable | None = None):
        """Print the solve report in the terminal in a table format

        Args:
            print_cal (Callable | None, optional): _description_. Defaults to None.
        """
        if print_cal is None:
            print_cal = print
        # Set column widths
        col1_width = 22  # Wider key column
        col2_width = 40  # Value column
        total_width = col1_width + col2_width + 5  # +5 for borders/padding

        def row(key, val):
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print_cal(f"| {key:<{col1_width}} | {val_str:<{col2_width}} |")  # ty: ignore

        border = "+" + "-" * (col1_width + 2) + "+" + "-" * (col2_width + 2) + "+"

        print_cal(border)
        print_cal(f"| {'FEM Solve Report':^{total_width - 2}} |")
        print_cal(border)
        row("Solver", self.solver)
        row("Sorter", self.sorter)
        row("Preconditioner", self.precon)
        row("Job ID", self.jobid)
        row("Sim Time (s)", self.simtime)
        row("DOFs (Total)", self.ndof)
        row("NNZ (Total)", self.nnz)
        row("DOFs (Solve)", self.ndof_solve)
        row("NNZ (Solve)", self.nnz_solve)
        row("Exit Code", self.exit_code)
        row("Worker", self.worker_name)
        print_cal(border)

        if self.aux:
            print_cal(f"| {'Additional Info':^{total_width - 2}} |")
            print_cal(border)
            for k, v in self.aux.items():
                row(str(k), v)
            print_cal(border)


def _pfx(name: str, id: int = 0) -> str:
    return f"[{name}-j{id:03d}]"


def is_numerically_complex_symmetric(A, rtol=0.025) -> tuple[float, bool]:
    num = np.linalg.norm((A - A.T).data)
    den = np.linalg.norm(A.data)
    ratio = num / den
    return ratio, ratio <= rtol


############################################################
#         COMPLEX MATRIX TO REAL MATRIX CONVERSION        #
############################################################


def complex_to_real_block(A, b):
    """Return (Â,  b̂) real-augmented representation of A x = b."""
    A_r = sparse.csc_matrix(A.real)
    A_i = sparse.csc_matrix(A.imag)
    #  [ ReA  -ImA ]
    #  [ ImA   ReA ]
    upper = sparse.hstack([A_r, -A_i])
    lower = sparse.hstack([A_i, A_r])
    A_hat = sparse.vstack([upper, lower]).tocsc()

    b_hat = np.hstack([b.real, b.imag])
    return A_hat, b_hat


def real_to_complex_block(x):
    """Return x = (x_r, x_i) as complex vector."""
    n = x.shape[0] // 2
    x_r = x[:n, :]
    x_i = x[n:, :]
    return x_r + 1j * x_i


############################################################
#                      SOLVERTYPEENUM                     #
############################################################


class SolverType(Enum):
    SINGLE_ONLY = 0  # One core, only one worker
    SINGLE_MT = 1  # One core, multiple workers multi-threading
    SINGLE_MP = 2  # One core, multiple workers multi-processing
    PARALLEL = 3  # Parallel (cannot work with multiple workers)


############################################################
#                  BASE CLASS DEFINITIONS                 #
############################################################


class SimulationError(Exception):
    pass


class Sorter:
    """A Generic class that executes a sort on the indices.
    It must implement a sort and unsort method.
    """

    def __init__(self):
        """Create a sorter without a permutation."""
        self.perm = None
        self.inv_perm = None

    def reset(self):
        """Reset the permuation vectors."""
        self.perm = None
        self.inv_perm = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def sort(
        self, A: csc_matrix, b: np.ndarray, reuse_sorting: bool = False
    ) -> tuple[csc_matrix, np.ndarray]:
        """Reorder the system matrix and right hand side.

        Args:
            A (csc_matrix): The system matrix
            b (np.ndarray): The right hand side
            reuse_sorting (bool, optional): Reuse the permutation of the previous call. Defaults to False.

        Returns:
            tuple[csc_matrix, np.ndarray]: The sorted matrix and right hand side
        """
        return A, b

    def unsort(self, x: np.ndarray) -> np.ndarray:
        """Undo the reordering on a solution vector.

        Args:
            x (np.ndarray): The solution of the sorted system

        Returns:
            np.ndarray: The solution in the original ordering
        """
        return x


class Preconditioner:
    """A Generic class defining a preconditioner as attribute .M based on the
    matrix A and b. This must be generated in the .init(A,b) method.
    """

    def __init__(self):
        self.M: np.ndarray = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def init(self, A: csc_matrix, b: np.ndarray) -> None:
        raise NotImplementedError("")


class Solver:
    """A generic class representing a solver for the problem Ax=b

    A solver class has two class attributes.
     - real_only: defines if the solver can only deal with real numbers. In this case
    the solve routine will automatically provide A and b in real number format.
     - req_sorter: defines if this solver requires the use of a sorter algorithm. By setting
     it to False, the SolveRoutine will not use the default sorting algorithm.
    """

    stype: SolverType = SolverType.SINGLE_ONLY
    name: str = "UNNAMED"
    real_only: bool = False
    req_sorter: bool = False
    released_gil: bool = False
    needs_precon_data: bool = False  # True if solve() needs a PreconData (set with set_precon_data)
    # [Written by Claude] Whether the library of this solver is installed on this system, and what to install
    available: ClassVar[bool] = True
    requires: ClassVar[str] = ""

    def __init__(self, pre: str = ""):
        """Create the solver.

        Args:
            pre (str, optional): The log message prefix. Defaults to "".
        """
        self.own_preconditioner: bool = False
        self.initialized: bool = False
        self.pre: str = pre

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def set_symmetry(self, complex_symmetric: bool) -> None:
        """Tell the solver whether the matrix is complex symmetric.

        Args:
            complex_symmetric (bool): Whether the matrix is complex symmetric
        """
        pass

    def initialize(self) -> None:
        """Initialize the solver backend (only done once)."""
        return None

    def duplicate(self) -> Solver:
        """Create a new solver of the same type with the same settings.

        Returns:
            Solver: The new solver
        """
        return self.__class__(self.pre)

    def set_options(self, pivoting_threshold: float | None = None) -> Solver:
        """Write generic simulation options to the solver object.
        Options may be ignored depending on the type of solver used.

        Args:
            pivoting_threshold (float | None, optional): The pivoting threshold of direct solvers. Defaults to None.
        """
        return self

    # [Written by Claude]
    def set_nthreads(self, nthreads: int | None) -> Solver:
        """Install the preferred number of CPU threads for the next solves.
        Solvers without a thread count implementation ignore it.

        Args:
            nthreads (int | None): The number of threads. None leaves the choice to the solver backend.
        """
        if nthreads is not None:
            logger.trace(f"{self.pre}{self} has no solve thread count implementation, ignoring nthreads={nthreads}.")
        return self

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
        precon: Preconditioner,
        reuse_factorization: bool = False,
        id: int = -1,
    ) -> tuple[np.ndarray, SolveReport]:
        """Solve the linear system Ax=b.

        Args:
            A (csc_matrix): The system matrix
            b (np.ndarray): The right hand side
            precon (Preconditioner): The preconditioner (used by iterative solvers)
            reuse_factorization (bool, optional): Reuse the factorization of a previous solve. Defaults to False.
            id (int, optional): The job id used in the log messages. Defaults to -1.

        Returns:
            tuple[np.ndarray, SolveReport]: The solution vector and the solve report
        """
        raise NotImplementedError("This classes Ax=B solver method is not implemented.")

    def reset(self) -> None:
        """Reset state variables like numeric and symbollic factorizations."""
        pass


class EigSolver:
    """A generic class representing a solver for the eigenvalue problem Ax=λBx

    A solver class has two class attributes.
     - real_only: defines if the solver can only deal with real numbers. In this case
    the solve routine will automatically provide A and b in real number format.
     - req_sorter: defines if this solver requires the use of a sorter algorithm. By setting
     it to False, the SolveRoutine will not use the default sorting algorithm.
    """

    name: str = "UNNAMED"
    real_only: bool = False
    req_sorter: bool = False
    # [Written by Claude] Whether the library of this solver is installed on this system, and what to install
    available: ClassVar[bool] = True
    requires: ClassVar[str] = ""

    def __init__(self, pre: str = ""):
        """Create the eigenvalue solver.

        Args:
            pre (str, optional): The log message prefix. Defaults to "".
        """
        self.own_preconditioner: bool = False
        self.pre: str = pre

    def initialize(self) -> None:
        """Initialize the solver backend (only done once)."""
        return None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def duplicate(self) -> Solver:
        """Create a new solver of the same type with the same settings.

        Returns:
            Solver: The new solver
        """
        return self.__class__(self.pre)

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int = 6,
        target_k0: float = 0.0,
        which: str = "LM",
        sign: float = 1.0,
    ):
        """Solve the generalized eigenvalue problem Ax=λBx.

        Args:
            A (csc_matrix): The A matrix
            B (csc_matrix): The B matrix
            nmodes (int, optional): The number of eigenmodes to find. Defaults to 6.
            target_k0 (float, optional): The k0 around which to search. Defaults to 0.0.
            which (str, optional): Which eigenvalues to find (ARPACK convention). Defaults to "LM".
            sign (float, optional): The sign applied to the shift. Defaults to 1.0.

        Returns:
            tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors
        """
        raise NotImplementedError(
            "This classes eigenmdoe solver method is not implemented."
        )

    def reset(self) -> None:
        """Reset state variables like numeric and symbollic factorizations."""
        pass
