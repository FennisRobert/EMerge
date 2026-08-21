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
from __future__ import annotations
from scipy.sparse import csc_matrix, save_npz# type: ignore
from scipy.sparse.csgraph import reverse_cuthill_mckee  # type: ignore
from scipy.sparse.linalg import bicgstab, cg, gmres, gcrotmk, eigs, splu  # type: ignore
from scipy.linalg import eig  # type: ignore
from scipy import sparse  # type: ignore
import hashlib
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
import platform
import time
from typing import Literal, Callable
from enum import Enum
from .file import Saveable
import os

############################################################
#                   ENVIRONMENT VARIABLES                  #
############################################################

# Environment variables are used to transfer settings from the
# main solver environment to parallel-processes that otherwise
# don't have access to these settings.

_FORCED_SOLVER = os.getenv("EMERGE_MP_SOLVER", default="")
_SYMMETRY_LIMIT = os.getenv("EMERGE_SYM_LIMIT", default="0.02")

############################################################
#                    SOLVER AVAILABILITY                   #
############################################################

_PARDISO_AVAILABLE = False
_UMFPACK_AVAILABLE = False
_CUDSS_AVAILABLE = False
_MUMPS_AVAILABLE = False
_AASDS_AVAILABLE = False
_SKSP_AVAILABLE = False

""" Check if the PC runs on a non-ARM architechture
If so, attempt to import PyPardiso (if its installed)
"""


############################################################
#                          PARDISO                         #
############################################################

if "arm" not in platform.processor():
    from .solve_interfaces.pardiso_interface import PardisoInterface

    _PARDISO_AVAILABLE = True


############################################################
#                          UMFPACK                         #
############################################################


try:
    import scikits.umfpack as um  # type: ignore

    _UMFPACK_AVAILABLE = True
except ModuleNotFoundError:
    logger.debug("UMFPACK not found, defaulting to SuperLU")

############################################################
#                           MUMPS                          #
############################################################

try:
    from .solve_interfaces.mumps_interface import MUMPSInterface  # type: ignore

    _MUMPS_AVAILABLE = True
except ModuleNotFoundError as e:
    logger.debug("MUMPS not found, defaulting to SuperLU")


############################################################
#                          AASDS                           #
############################################################
try:
    from .solve_interfaces.aasds_interface import AASDSInterface  # type: ignore

    _AASDS_AVAILABLE = True
except ModuleNotFoundError as e:
    logger.debug(e)
    logger.debug("AASDS not found, defaulting to SuperLU")
except ImportError as e:
    logger.debug(e)
    logger.debug("Tried to import an installed AASDS solver on a non Darwin system.")

############################################################
#                           CUDSS                          #
############################################################


try:
    from .solve_interfaces.cudss_interface import CuDSSInterface

    _CUDSS_AVAILABLE = True
except ModuleNotFoundError:
    pass
except ImportError as e:
    logger.error("Error while importing CuDSS dependencies:")
    logger.exception(e)


############################################################
#                       SCIKIT SPARSE                      #
############################################################

try:
    from sksparse.cholmod import cho_factor, metis

    _SKSP_AVAILABLE = True
except ModuleNotFoundError:
    pass


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
#                 EIGENMODE FILTER ROUTINE                #
############################################################


def filter_real_modes(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    k0: float,
    ermax: complex,
    urmax: complex,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filters eigenmodes that are actually real (removing spurious ones generated by ARPACK)

    Uses the eigen values constrianed by a search space limited by the
    propagation constant.

    Args:
        eigvals (np.ndarray): array of eigenvalues
        eigvecs (np.ndarray): 2D array of eigenvectors
        k0 (float): Propagation constant
        ermax (complex): Maximum epsilon_r
        urmax (complex): maximum mu_r
        sign (float): expected sign before eigenvalue (can be configured externally)

    Returns:
        tuple[np.ndarray, np.ndarray]: A filtered list of unique eigenvalues/vectors
    """
    minimum = 1
    extremum = (k0**2) * ermax * urmax * 2
    eigvals[eigvals == np.inf] = 1e30
    eigvals[eigvals == -np.inf] = -1e30
    mask = (sign * eigvals <= extremum) & (sign * eigvals >= minimum)
    filtered_vals = eigvals[mask]
    filtered_vecs = eigvecs[:, mask]
    k0vals = np.sqrt(sign * filtered_vals)
    order = np.argsort(np.abs(k0vals))  # ascending distance
    filtered_vals = filtered_vals[order]  # reorder eigenvalues
    filtered_vecs = filtered_vecs[:, order]
    return filtered_vals, filtered_vecs


############################################################
#               EIGENMODE ORTHOGONALITY CHECK              #
############################################################


def filter_unique_eigenpairs(
    eigen_values: list[complex], eigen_vectors: list[np.ndarray], tol=-3
) -> tuple[list[complex], list[np.ndarray]]:
    """
    Filters eigenvectors by orthogonality using dot-product tolerance.

    Parameters:
        eigen_values (np.ndarray): Array of eigenvalues, shape (n,)
        eigen_vectors (np.ndarray): Array of eigenvectors, shape (n, n)
        tol (float): Dot product tolerance for considering vectors orthogonal (default: 1e-5)

    Returns:
        unique_values (np.ndarray): Filtered eigenvalues
        unique_vectors (np.ndarray): Corresponding orthogonal eigenvectors
    """
    if len(eigen_values) <= 1:
        return eigen_values, eigen_vectors

    selected: list = [eigen_vectors[0] / np.linalg.norm(eigen_vectors[0])]
    indices: list = [0]
    for i in range(1, len(eigen_vectors)):
        vec = eigen_vectors[i]
        vec = vec / np.linalg.norm(vec)  # Normalize

        tols = [10 * np.log10(abs(np.dot(vec, sel))) for sel in selected]
        if all(t < tol for t in tols):
            selected.append(vec)
            indices.append(i)

    unique_values = [eigen_values[i] for i in indices]
    unique_vectors = [eigen_vectors[i] for i in indices]

    return unique_values, unique_vectors


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
        return A, b

    def unsort(self, x: np.ndarray) -> np.ndarray:
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

    def __init__(self, pre: str = ""):
        self.own_preconditioner: bool = False
        self.initialized: bool = False
        self.pre: str = pre

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def set_symmetry(self, complex_symmetric: bool) -> None:
        pass

    def initialize(self) -> None:
        return None

    def duplicate(self) -> Solver:
        return self.__class__(self.pre)

    def set_options(self, pivoting_threshold: float | None = None) -> None:
        """Write generic simulation options to the solver object.
        Options may be ignored depending on the type of solver used."""
        pass

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
        precon: Preconditioner,
        reuse_factorization: bool = False,
        id: int = -1,
    ) -> tuple[np.ndarray, SolveReport]:
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

    def __init__(self, pre: str = ""):
        self.own_preconditioner: bool = False
        self.pre: str = pre

    def initialize(self) -> None:
        return None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def duplicate(self) -> Solver:
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
        raise NotImplementedError(
            "This classes eigenmdoe solver method is not implemented."
        )

    def reset(self) -> None:
        """Reset state variables like numeric and symbollic factorizations."""
        pass


############################################################
#                          SORTERS                         #
############################################################


class ReverseCuthillMckee(Sorter):
    """Implements the Reverse Cuthill-Mckee sorting."""

    def __init__(self):
        super().__init__()

    def sort(self, A, b, reuse_sorting: bool = False):
        if not reuse_sorting:
            logger.debug("Generating Reverse Cuthill-Mckee sorting.")
            self.perm = reverse_cuthill_mckee(A)
            self.inv_perm = np.argsort(self.perm)
        logger.debug("Applying Reverse Cuthill-Mckee sorting.")
        Asorted = A[self.perm, :][:, self.perm]
        bsorted = b[self.perm]
        return Asorted, bsorted

    def unsort(self, x: np.ndarray):
        logger.debug("Reversing Reverse Cuthill-Mckee sorting.")
        return x[self.inv_perm]


############################################################
#                      PRECONDITIONERS                     #
############################################################


class ILUPrecon(Preconditioner):
    """Implements the incomplete LU preconditioner on matrix A."""

    def __init__(self):
        super().__init__()
        self.M = None
        self.fill_factor = 20
        self.options: dict[str, str] = dict(SymmetricMode=True)

    def init(self, A, b):
        logger.info("Generating ILU Preconditioner")
        self.ilu = sparse.linalg.spilu(
            A,
            drop_tol=1e-3,
            drop_rule="basic,area,interp",
            fill_factor=self.fill_factor,  # ty: ignore
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.001,
            options=self.options,
        )
        self.M = sparse.linalg.LinearOperator(A.shape, self.ilu.solve)  # ty: ignore


class JacobiPrecon(Preconditioner):
    """Implements the diagonal (Jacobi) preconditioner on matrix A."""

    def __init__(self):
        super().__init__()
        self.M = None

    def init(self, A, b):
        logger.info("Generating Jacobi Preconditioner")
        diag_inv = 1.0 / A.diagonal()
        self.M = sparse.linalg.LinearOperator(A.shape, lambda x: diag_inv * x)


class ICCPrecon(Preconditioner):
    def __init__(self):
        super().__init__()
        self.M = None
        self.beta = 0.0

    def init(self, A, b):
        logger.info(f"Generating ICC Preconditioner (beta={self.beta})")
        factor = cho_factor(A, order="colamd", beta=self.beta)
        self.M = sparse.linalg.LinearOperator(A.shape, factor.solve)


############################################################
#                     ITERATIVE SOLVERS                    #
############################################################


class SolverCG(Solver):
    """Implements the Conjugate Gradient method for SPD systems"""

    req_sorter = True
    name = "CG"
    stype = SolverType.SINGLE_ONLY

    def __init__(self, pre: str):
        super().__init__(pre)
        self.atol = 1e-10

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        self._iter += 1
        if self._iter % 50 == 0:
            res = np.linalg.norm(self.A @ xk - self.b.ravel())
            logger.info(f"{self.pre}: Iteration {self._iter}, |r| = {res:.2e}")

    def solve(self, A, b, precon, reuse_factorization=False, id=-1):
        logger.info(f"{_pfx(self.pre, id)} Calling CG.")
        self._iter = 0
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = cg(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = cg(A, b, atol=self.atol, callback=self.callback)
        logger.info(f"{self.pre}: Converged in {self._iter} iterations.")
        x = x.reshape(b.shape)
        return x, SolveReport(solver=str(self), exit_code=info)


class SolverBicgstab(Solver):
    """Implements the Bi-Conjugate Gradient Stabilized method"""

    req_sorter = True
    name = "BISCGSTAB"
    stype = SolverType.SINGLE_ONLY

    def __init__(self, pre: str):
        super().__init__(pre)
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        self._iter += 1
        if self._iter % 10 == 0:
            logger.info(f"{self.pre}: Iteration {self._iter}")

    def solve(
        self, A, b, precon, reuse_factorization: bool = False, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling BiCGStab.")
        self._iter = 0
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = bicgstab(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = bicgstab(A, b, atol=self.atol, callback=self.callback)
        logger.info(f"{self.pre}: Converged in {self._iter} iterations.")
        x = x.reshape(b.shape)
        return x, SolveReport(solver=str(self), exit_code=info)


class SolverGCROTMK(Solver):
    stype = SolverType.SINGLE_ONLY
    name: str = "GCROTMK"
    """ Implements the GCRO-T(m,k) Iterative solver. """

    def __init__(self):
        super().__init__()
        self.atol = 1e-5
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(self.pre + f"Iteration {convergence:.4f}")

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
        precon: Preconditioner,
        reuse_factorization: bool = False,
        id: int = -1,
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling GCRO-T(m,k) algorithm")
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gcrotmk(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = gcrotmk(A, b, atol=self.atol, callback=self.callback)
        return x, SolveReport(solver=str(self), exit_code=info)


class SolverGMRES(Solver):
    """Implements the GMRES solver."""

    stype = SolverType.SINGLE_ONLY
    real_only = False
    req_sorter = True
    name = "GMRES"

    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, norm):
        # convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(self.pre + f"Iteration {norm:.4f}")

    def solve(
        self, A, b, precon, reuse_factorization: bool = False, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling GMRES Function.")
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gmres(
                A,
                b,
                M=precon.M,
                atol=self.atol,
                callback=self.callback,
                callback_type="pr_norm",
            )
        else:
            x, info = gmres(
                A,
                b,
                atol=self.atol,
                callback=self.callback,
                restart=500,
                callback_type="pr_norm",
            )
        return x, SolveReport(solver=str(self), exit_code=info)


############################################################
#                      DIRECT SOLVERS                     #
############################################################


class SolverCHOLMOD(Solver):
    """CHOLMOD direct solver with nested dissection ordering for SPD systems."""

    req_sorter = False
    stype = SolverType.SINGLE_MT
    name = "CHOLMOD_ND"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.factor = None
        self.perm = None

    def reset(self):
        self.factor = None
        self.perm = None

    def duplicate(self):
        return self.__class__(self.pre)

    def solve(self, A, b, precon, reuse_factorization=False, id=-1):
        logger.info(f"{_pfx(self.pre, id)} Calling CHOLMOD (Nested Dissection) Solver.")

        if b.ndim == 1:
            b = b.reshape(-1, 1)

        if not reuse_factorization or self.factor is None:
            logger.trace(f"{_pfx(self.pre, id)} Computing Cholesky factorization.")
            self.factor = cho_factor(
                A, order="colamd", sym_kind="sym", supernodal_mode="supernodal"
            )

        x = self.factor.solve(b)

        return x, SolveReport(solver=str(self), exit_code=0)


class SolverSuperLU(Solver):
    """Implements Scipy's direct SuperLU solver, with optional METIS preordering."""

    req_sorter: bool = False
    real_only: bool = False
    released_gil: bool = True
    stype = SolverType.SINGLE_MT
    name = "SUPERLU"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.atol = 1e-5
        self.A = None
        self.b = None
        self.options: dict[str, str] = dict(
            SymmetricMode=True, Equil=False, IterRefine="SINGLE"
        )
        self._pivoting_threshold: float = 0.001
        self.lu = None
        self._perm = None
        self._a_hash: bytes = None

    def hash_A(self, A: csc_matrix) -> bytes:
        """Hashes the A matrix"""
        hasher = hashlib.sha256()
        hasher.update(A.indptr.tobytes())
        hasher.update(A.indices.tobytes())
        self._a_hash = hasher.digest()
        return self._a_hash

    def reset(self):
        self.A = None
        self.b = None
        self.lu = None
        self._perm = None
        self._a_hash = None
        self.options = dict(SymmetricMode=True, Equil=False, IterRefine="SINGLE")
        self._pivoting_threshold = 0.001

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        new_solver._pivoting_threshold = self._pivoting_threshold
        return new_solver

    def set_options(self, pivoting_threshold: float | None = None) -> None:
        if pivoting_threshold is not None:
            self._pivoting_threshold = pivoting_threshold

    def _metis(self, A: csc_matrix, hash_value: bytes, idf: int) -> None:
        """Executes METIS column preordering algorithm from Scikit-Sparse

        Args:
            A (csc_matrix): The A-matrix
            hash_value (bytes): _description_
        """
        logger.debug(f"{_pfx(self.pre, idf)} Computing METIS permutation.")
        self._perm = metis(A)
        self._a_hash = hash_value

    def _metis_if_needed(self, A: csc_matrix, idf: int) -> None:
        """Executes a new METIS column preordering only if there is no
        ordering or the ordering is of a different matrix.

        Args:
            A (csc_matrix): _description_
            idf (int): _description_
        """
        hash_A = self.hash_A(A)
        if self._perm is None:
            logger.trace("No self._perm, requiring new METIS ordering.")
            self._metis(A, hash_A, idf)
            return
        if hash_A != self._a_hash:
            logger.trace("Different hash detected, requiring new METIS ordering.")
            self._metis(A, hash_A, idf)

    def solve(
        self, A: csc_matrix, b: np.ndarray, precon, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling SuperLU Solver.")

        if _SKSP_AVAILABLE:
            self._metis_if_needed(A, id)
            A_ordered = A[self._perm][:, self._perm]
            permc = "NATURAL"
        else:
            self._perm = None
            A_ordered = A
            permc = "MMD_AT_PLUS_A"

        logger.trace(f"{_pfx(self.pre, id)} Computing LU-Decomposition.")
        self.lu = splu(
            A_ordered,
            permc_spec=permc,
            relax=0,
            diag_pivot_thresh=self._pivoting_threshold,
            options=self.options,
        )

        if self._perm is not None:
            if b.ndim == 1:
                bp = b[self._perm]
                xp = self.lu.solve(bp)
                x = np.empty_like(xp)
                x[self._perm] = xp
            else:
                x = np.empty_like(b)
                for i in range(b.shape[1]):
                    xp = self.lu.solve(b[self._perm, i])
                    x[self._perm, i] = xp
        else:
            x = self.lu.solve(b)

        aux = {"pivoting threshold": str(self._pivoting_threshold)}
        return x, SolveReport(solver=str(self), exit_code=0, aux=aux)


class SolverUMFPACK(Solver):
    """Implements the UMFPACK Sparse SP solver."""

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "UMFPACK"

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating UMFPACK solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None

        self.umfpack: um.UmfpackContext | None = None

        # SETTINGS
        self._pivoting_threshold: float = 0.001

        self.fact_symb: bool = False
        self.initalized: bool = False

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing UMFPACK Solver")
        self.umfpack = um.UmfpackContext("zl")
        self.umfpack.control[um.UMFPACK_PRL] = 0  # ty: ignore
        self.umfpack.control[um.UMFPACK_IRSTEP] = 2  # ty: ignore
        self.umfpack.control[um.UMFPACK_STRATEGY] = um.UMFPACK_STRATEGY_SYMMETRIC  # ty: ignore
        self.umfpack.control[um.UMFPACK_ORDERING] = 3  # ty: ignore
        self.umfpack.control[um.UMFPACK_PIVOT_TOLERANCE] = 0.001  # ty: ignore
        self.umfpack.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = 0.001  # ty: ignore
        self.umfpack.control[um.UMFPACK_BLOCK_SIZE] = 64  # ty: ignore
        self.umfpack.control[um.UMFPACK_FIXQ] = -1  # ty: ignore
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting UMFPACK solver state")
        self.fact_symb = False

    def set_options(self, pivoting_threshold: float | None = None) -> None:
        self.initialize()
        if pivoting_threshold is not None:
            self.umfpack.control[um.UMFPACK_PIVOT_TOLERANCE] = pivoting_threshold  # ty: ignore
            self.umfpack.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = pivoting_threshold  # ty: ignore
            self._pivoting_threshold = pivoting_threshold

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        new_solver.set_options(pivoting_threshold=self._pivoting_threshold)
        return new_solver

    def solve(self, A, b, precon, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling UMFPACK Solver.")
        A.indptr = A.indptr.astype(np.int64)
        A.indices = A.indices.astype(np.int64)
        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Executing symbollic factorization.")
            self.umfpack.symbolic(A)
            self.fact_symb = True

        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        self.umfpack.numeric(A)
        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(b.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solving RHS {i}")
            x[:, i] = self.umfpack.solve(um.UMFPACK_A, A, b[:, i], autoTranspose=False)  # ty: ignore
        aux = {"Pivoting Threshold": str(self._pivoting_threshold)}
        return x, SolveReport(solver=str(self), exit_code=0, aux=aux)


class SolverMUMPS(Solver):
    """Implements the MUMPS Sparse SP solver."""

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating MUMPS solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None

        self.mumps: MUMPSInterface | None = None

        # SETTINGS
        self._pivoting_threshold: float = 0.001

        self.fact_symb: bool = False
        self.initalized: bool = False

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + "Initializing MUMPS Solver")
        self.mumps = MUMPSInterface()
        self.initalized = True

    def reset(self) -> None:
        logger.trace(self.pre + "Resetting MUMPS solver state")
        self.fact_symb = False

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        return new_solver

    def solve(
        self, A, b, precon, reuse_factorization: bool = False, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling MUMPS Solver.")
        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Executing symbollic factorization.")
            self.mumps.analyse_matrix(A)
            self.fact_symb = True
        if not reuse_factorization:
            logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
            self.mumps.factorize(A)
            self.A = A
        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(x.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solve RHS {i}.")
            x[:, i], _ = self.mumps.solve(b[:, i])  # ty: ignore
        return x, SolveReport(solver=str(self), exit_code=0)


class SolverAASDS(Solver):
    """Implements the Apple Accelerate Sparse Direct solver."""

    req_sorter = False
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "AASDS"

    def __init__(self, pre: str):
        super().__init__(pre)
        logger.trace(self.pre + "Creating Apple Accelerate solver")
        self.A: np.ndarray = None
        self.b: np.ndarray = None

        self.aasds: AASDSInterface | None = None
        self._csym: bool = True
        self.sym_fact: str = 'lu'   # Ensures backwards compatibility with older installations of emerge-aasds
        self.unsym_fact: str = 'lu' # Ensures backwards compatibility with older installations of emerge-aasds

        # SETTINGS
        self._pivoting_threshold: float = 0.001

        self.fact_symb: bool = False
        self.initalized: bool = False

    def set_symmetry(self, complex_symmetric: bool) -> None:
        self.aasds._csym = complex_symmetric

    def set_factorization(self, sym_fact: Literal['LU','LUUP','LUSPP','LUTPP','LDLT','LDLTUP','LDLTSBK','LDLTTPP'] = 'LU',
                          unsym_fact: Literal['LU','LUUP','LUSPP','LUTPP','LDLT','LDLTUP','LDLTSBK','LDLTTPP'] = 'LU'):
        """Defines the factorization algorithm:

        Options:
            'LU'      = Default LU factorization.
            'LUUP'    = LU factorization Unpivoted (no numerical pivoting).
            'LUSPP'   = LU factorization with Supernode Partial Pivoting (restricted to within supernodes).
            'LUTPP'   = LU factorization with Threshold Partial Pivoting.
            'LDLT'    = Default LDLᵀ factorization (for symmetric matrices).
            'LDLTUP'  = LDLᵀ factorization Unpivoted (Cholesky-like, no pivoting).
            'LDLTSBK' = LDLᵀ factorization with Supernode-Bunch-Kaufman and static pivoting.
            'LDLTTPP' = LDLᵀ factorization with full-threshold partial pivoting.

        Args:
            sym_fact (str): Factorization algorithm for Symmetric matrices
            unsym_fact (str): Factorization algorithm for Unsymmetric matrices
        """
        unsym_options = ['LU','LUUP','LUSPP','LUTPP']

        self.sym_fact = sym_fact.lower()
        if unsym_fact not in unsym_options:
            logger.warning('Unsymmetric matrices cannot ude LDLT algorithms. Defaulting to LU')
            self.unsym_fact = 'lu'
        else:
            self.unsym_fact = unsym_fact.lower()

        self.initialize()
        self.configure()

    def initialize(self):
        if self.initalized:
            return
        logger.trace(self.pre + " Initializing Apple Accelerate Solver")
        self.aasds = AASDSInterface()
        self.configure()
        self.initalized = True

    def configure(self):
        self.aasds._factorization = self.aasds._fact(self.sym_fact)
        self.aasds._usfactorization = self.aasds._fact(self.unsym_fact)

    def reset(self) -> None:
        logger.trace(self.pre + " Resetting Apple Accelerate solver state")
        self.fact_symb = False
        self._csym = False

    def duplicate(self) -> Solver:
        new_solver = self.__class__(self.pre)
        return new_solver

    def solve(
        self, A: csc_matrix, b, precon, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:
        logger.info(f"{_pfx(self.pre, id)} Calling Accelerate Solver.")
        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Executing symbolic factorization.")
            self.aasds.analyse(A)
            self.fact_symb = True
        if True:
            logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
            self.aasds.factorize(A)
            self.A = A
        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(x.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solve RHS {i}.")
            x[:, i], _ = self.aasds.solve(b[:, i])

        #
        return x, SolveReport(solver=str(self), exit_code=0)


class SolverPardiso(Solver):
    """Implements the PARDISO solver through PyPardiso."""

    real_only: bool = False
    req_sorter: bool = False
    stype = SolverType.PARALLEL
    name = "PARDISO"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.solver: PardisoInterface | None = None
        self.fact_symb: bool = False
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def initialize(self) -> None:
        if self.initialized:
            return
        self.solver = PardisoInterface()
        self.initialized = True

    def reset(self) -> None:
        self.A = None
        self.B = None
        if self.solver is not None:
            self.solver.clear_memory()

    def solve(
        self, A: csc_matrix, b, precon, id: int = -1
    ) -> tuple[np.ndarray, SolveReport]:

        logger.info(f"{_pfx(self.pre, id)} Calling Pardiso Solver")
        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Executing symbollic factorization.")
            self.solver.symbolic(A)
            self.fact_symb = True

        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        self.solver.numeric(A)
        self.A = A

        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(x.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solving RHS {i}")
            x[:, i], error = self.solver.solve(A, b[:, i])

        if error != 0:
            logger.error(f"{_pfx(self.pre, id)} Terminated with error code {error}")
            logger.error(self.pre + self.solver.get_error(error))
            raise SimulationError(
                f"{_pfx(self.pre, id)} PARDISO Terminated with error code {error}"
            )

        aux = {}
        return x, SolveReport(solver=str(self), exit_code=error, aux=aux)


class SolverCuDSS(Solver):
    real_only = False
    stype = SolverType.SINGLE_ONLY
    name = "CUDSS"

    def __init__(self, pre: str):
        super().__init__(pre)
        self._cudss: CuDSSInterface | None = None
        self.fact_symb: bool = False
        self.fact_numb: bool = False
        self._csym: bool = True

    def initialize(self) -> None:
        if self.initialized:
            return
        self._cudss = CuDSSInterface()
        self._cudss._PRES = 2
        self.initialized = True

    def set_symmetry(self, complex_symmetric: bool) -> None:
        self._cudss._csym = complex_symmetric

    def reset(self) -> None:
        self.fact_symb = False
        self.fact_numb = False
        if self._cudss is not None:
            self._cudss.clear_memory()

    def solve(self, A, b, precon, id: int = -1):
        logger.info(f"{_pfx(self.pre, id)} Calling cuDSS Solver")

        if self.fact_symb is False:
            logger.trace(f"{_pfx(self.pre, id)} Starting from symbollic factorization.")
            x = self._cudss.symbolic(A)
            self.fact_symb = True

        logger.trace(f"{_pfx(self.pre, id)} Starting from numeric factorization.")
        x = self._cudss.numeric(A)
        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(b.shape[1]):
            x[:, i] = self._cudss.solve(b[:, i])

        return x, SolveReport(solver=str(self), exit_code=0, aux={})


############################################################
#                 DIRECT EIGENMODE SOLVERS                #
############################################################


class SolverLAPACK(EigSolver):
    name = "LAPACK"

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int = 6,
        target_k0: float = 0,
        which: str = "LM",
        sign: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dense solver for  A x = λ B x   with A = Aᴴ, B = Bᴴ (B may be indefinite).

        Parameters
        ----------
        A, B : (n, n) array_like, complex127/complex64/float64
        k    : int or None
            How many eigenpairs to return.
            * None  → return all n
            * k>0   → return k pairs with |λ| smallest

        Returns
        -------
        lam  : (m,) real ndarray      eigenvalues  (m = n or k)
        vecs : (n, m) complex ndarray eigenvectors, B-orthonormal  (xiᴴ B xj = δij)
        """
        logger.info(f"{_pfx(self.pre)} Calling LAPACK eig solver")
        lam, vecs = eig(
            A.toarray(),
            B.toarray(),
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
        )
        lam, vecs = filter_real_modes(lam, vecs, target_k0, 2, 2, sign=sign)
        return lam, vecs


############################################################
#                  ITERATIVE EIGEN SOLVERS                 #
############################################################


class SolverARPACK(EigSolver):
    """Implements the Scipy ARPACK iterative eigenmode solver."""

    name = "ARPACK"

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int = 6,
        target_k0: float = 0,
        which: str = "LM",
        sign: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"{_pfx(self.pre)} Searching for {nmodes} modes around β = {target_k0:.2f} rad/m mode={which} with ARPACK"
        )
        sigma = (sign * (target_k0**2)).real
        eigen_values, eigen_modes = eigs(A, k=nmodes, M=B, sigma=sigma, which=which)
        return eigen_values, eigen_modes


class SmartARPACK_BMA(EigSolver):
    """Implements the Scipy ARPACK iterative eigenmode solver with automatic search.

    The Solver searches in a geometric range around the target wave constant.
    """

    name = "ARPACKBMA"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.symmetric_steps: int = 41
        self.search_range: float = 2.0
        self.energy_limit: float = 1e-4
        self.ratio_limit: float = 1e-2

        self.tot_eigen_values: list[complex] = []
        self.tot_eigen_modes: list[np.ndarray] = []
        self.tot_energies: list[float] = []

    @property
    def n_found_modes(self) -> int:
        return len(self.tot_eigen_values)

    @staticmethod
    def are_modes_identical(
        v_old: np.ndarray, v_new: np.ndarray, tolerance: float = 0.99
    ) -> bool:
        """
        Check if two complex modes are the same regardless of phase or amplitude.

        Parameters:
            v_old, v_new: 1D arrays of complex degrees of freedom.
            tolerance: Threshold close to 1.0 (e.g., 0.99 or 0.999 for numerical matches).
        """
        # Compute the complex inner product (Hermitian dot product)
        inner_product = np.dot(np.conj(v_new), v_old)

        # Compute self-products (magnitudes squared)
        mag_new = np.dot(np.conj(v_new), v_new).real
        mag_old = np.dot(np.conj(v_old), v_old).real

        # Calculate CMAC
        cmac = (np.abs(inner_product) ** 2) / (mag_new * mag_old)

        logger.trace(f"    Modal Similarity (CMAC): {cmac:.5f}")
        return cmac >= tolerance

    def add_mode(
        self, eigen_value: complex, eigen_vector: np.ndarray, energy: float
    ) -> None:
        logger.debug(
            f"  Considering new eigenmode with value {eigen_value} and energy {energy:.4f}"
        )
        if self.n_found_modes == 0:
            logger.trace("    adding new mode")
            self.tot_eigen_values.append(eigen_value)
            self.tot_eigen_modes.append(eigen_vector)
            self.tot_energies.append(energy)
            return

        for mode in self.tot_eigen_modes:
            if self.are_modes_identical(mode, eigen_vector):
                continue
            logger.trace("    adding new mode")
            self.tot_eigen_values.append(eigen_value)
            self.tot_eigen_modes.append(eigen_vector)
            self.tot_energies.append(energy)
            return
        logger.trace("    ignoring mode because its the same as an existing mode")

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int = 6,
        target_k0: float = 0,
        which: str = "LM",
        sign: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:

        logger.info(
            f"{_pfx(self.pre)} Searching around β = {target_k0:.2f} rad/m with SmartARPACK (BMA)"
        )

        # ARPACK uses a parameter which is the expected eigenvalue to search around.
        # Because the exact eigenvalue is unknown, this scaler is used to scale
        # target_k0
        eig_search_scaler = np.geomspace(1, self.search_range, self.symmetric_steps)

        # cache eigenvalues
        self.tot_eigen_values = []
        self.tot_eigen_modes = []
        self.tot_energies = []

        # The Curl vs Total energy ratio should be in the order of k0**2 so keeping a safe margin:
        ratio_limit = (0.1 * target_k0) ** 2

        q_factors = []
        for q in eig_search_scaler:
            q_factors.append(q)
            q_factors.append(1 / q)
        q_factors.pop(0)  # 1 and 1/1 is redundant

        # Search around k_0
        for i, q in enumerate(q_factors):
            # Search around q*k0
            sigma = sign * ((q * target_k0) ** 2)
            logger.trace(f" Searching around {q * target_k0:.2f} rad/m")

            n_search = nmodes - self.n_found_modes
            eigen_values, eigen_modes = eigs(
                A, k=n_search, M=B, sigma=sigma, which=which
            )
            for i_sol in range(n_search):
                eigen_mode = eigen_modes[:, i_sol]
                eigen_value = eigen_values[i_sol]
                # Compute the energy
                energy = np.mean(np.abs(eigen_mode) ** 2)

                # Compute curl and total energy
                curl_energy = np.real(eigen_mode.conj() @ (A @ eigen_mode))
                total_energy = np.real(eigen_mode.conj() @ (B @ eigen_mode))
                ratio = abs(curl_energy / (total_energy + 1e-15))
                logger.debug(
                    f"Ratio = {ratio:.6f}, Energy = {energy:.4f}, value = {(sign * eigen_value) ** 0.5}, curl_energy = {curl_energy}, total_energy = {total_energy}"
                )

                if ratio > ratio_limit and energy > self.energy_limit and total_energy > 0.1:
                    self.add_mode(eigen_value, eigen_mode, energy)

                # Break if you found enough valid modes.
                if self.n_found_modes >= nmodes:
                    break

            # Break if you found enough valid modes.
            if self.n_found_modes >= nmodes:
                break

        # Sort solutions on mode energy
        if not self.tot_eigen_values:
            return np.array([]), np.array([])
        val, mode, energy = zip(
            *sorted(
                zip(self.tot_eigen_values, self.tot_eigen_modes, self.tot_energies),
                key=lambda x: x[2],
                reverse=True,
            )
        )
        eigen_values, eigen_modes = filter_unique_eigenpairs(val, mode)

        eigen_values = np.array(eigen_values)
        eigen_modes = np.array(eigen_modes).T

        return eigen_values, eigen_modes


class SmartARPACK(EigSolver):
    """Implements the Scipy ARPACK iterative eigenmode solver with automatic search.

    The Solver searches in a geometric range around the target wave constant.
    """

    name = "SMARTARPACK"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.symmetric_steps: int = 3
        self.search_range: float = 2.0
        self.energy_limit: float = 1e-4

    def reduce_solutions(
        self,
        A: csc_matrix,
        B: csc_matrix,
        eigen_values: np.ndarray,
        eigen_modes: np.ndarray,
        limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduces the solutions by filtering out modes that don't have
        an appropriate ratio of the curl term energy and total energy.

        Args:
            A (csc_matrix): _description_
            B (csc_matrix): _description_
            eigen_values (np.ndarray): _description_
            eigen_modes (np.ndarray): _description_
            limit (float): _description_

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        N = eigen_values.shape[0]
        eigen_values_out = []
        eigen_modes_out = []
        for i in range(N):
            value = eigen_values[i]
            mode = eigen_modes[:, i]
            energy = np.mean(np.abs(mode) ** 2)
            curl_energy = np.real(mode.conj() @ (A @ mode))
            total_energy = np.real(mode @ (B @ mode))
            ratio = abs(curl_energy / (total_energy + 1e-15))
            if ratio > limit and energy > self.energy_limit:
                eigen_values_out.append(value)
                eigen_modes_out.append(mode)
        return eigen_values_out, eigen_modes_out

    def eig(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int = 6,
        target_k0: float = 0,
        which: str = "LM",
        sign: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"{_pfx(self.pre)} Searching around 	β = {target_k0:.2f} rad/m with SmartARPACK"
        )
        # For BMA we can use a more constrained search space
        eig_search_scalers = [1.0, 0.99, 1.01, 0.95, 1.05, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
        tot_eigen_values = []
        tot_eigen_modes = []

        ratio_limit = 0.1 * target_k0**2

        for i, q in enumerate(eig_search_scalers):
            logger.trace(
                f"{_pfx(self.pre)} Modes Found = {len(tot_eigen_values)}, Search ratio: {q}"
            )
            sigma = sign * ((q * target_k0) ** 2)
            eigen_values, eigen_modes = eigs(A, k=nmodes, M=B, sigma=sigma, which=which)
            eigen_values, eigen_modes = self.reduce_solutions(
                A, B, eigen_values, eigen_modes, ratio_limit
            )
            tot_eigen_values.extend(eigen_values)
            tot_eigen_modes.extend(eigen_modes)
            tot_eigen_values, tot_eigen_modes = filter_unique_eigenpairs(
                tot_eigen_values, tot_eigen_modes
            )
            if len(tot_eigen_values) >= nmodes:
                break

        # Sort solutions on mode energy
        val, mode = tot_eigen_values, tot_eigen_modes
        val, mode = zip(*sorted(zip(val, mode), key=lambda x: x[0], reverse=False))  # type: ignore
        eigen_values = np.array(val[:nmodes])
        eigen_modes = np.array(mode[:nmodes]).T

        return eigen_values, eigen_modes


############################################################
#                        SOLVER ENUM                       #
############################################################


class EMSolver(Enum):
    SUPERLU = 1
    UMFPACK = 2
    PARDISO = 3
    LAPACK = 4
    ARPACK = 5
    SMART_ARPACK = 6
    SMART_ARPACK_BMA = 7
    CUDSS = 8
    MUMPS = 9
    AASDS = 10
    BICGSTAB = 11
    CG = 12
    CHOLMOD = 13

    def create_solver(self, pre: str) -> Solver | EigSolver | None:
        """Create a solver class instance or None if the solver is not available."""
        if self == EMSolver.UMFPACK and not _UMFPACK_AVAILABLE:
            return None
        elif self == EMSolver.AASDS and not _AASDS_AVAILABLE:
            return None
        elif self == EMSolver.PARDISO and not _PARDISO_AVAILABLE:
            return None
        elif self == EMSolver.MUMPS and not _MUMPS_AVAILABLE:
            return None
        if self == EMSolver.CUDSS and not _CUDSS_AVAILABLE:
            return None
        if self == EMSolver.CHOLMOD and not _SKSP_AVAILABLE:
            return None
        return self._clss(pre)

    @property
    def _clss(self) -> type[Solver]:
        mapper = {
            1: SolverSuperLU,
            2: SolverUMFPACK,
            3: SolverPardiso,
            4: SolverLAPACK,
            5: SolverARPACK,
            6: SmartARPACK,
            7: SmartARPACK_BMA,
            8: SolverCuDSS,
            9: SolverMUMPS,
            10: SolverAASDS,
            11: SolverBicgstab,
            12: SolverCG,
            13: SolverCHOLMOD,
        }
        return mapper.get(self.value, None)

    def istype(self, solver: Solver) -> bool:
        return isinstance(solver, self._clss)


############################################################
#                       SOLVE ROUTINE                      #
############################################################


class SolveRoutine:
    """A generic class describing a solve routine.
    A solve routine contains all the relevant sorter preconditioner and solver objects
    and goes through a sequence of steps to solve a linear system or find eigenmodes.

    """

    def __init__(self, thread_nr: int = 0, proc_nr: int = 0):

        self.pre: str = ""
        self._set_name(thread_nr, proc_nr)

        self.sorter: Sorter = ReverseCuthillMckee()
        self.precon: Preconditioner = JacobiPrecon()
        self.solvers: dict[EMSolver, Solver | EigSolver] = {
            slv: slv.create_solver(self.pre) for slv in EMSolver
        }
        self.solvers = {
            key: solver for key, solver in self.solvers.items() if solver is not None
        }

        self.parallel: Literal["SI", "MT", "MP"] = "SI"
        self.smart_search: bool = False
        self.forced_solver: list[Solver | EigSolver] = []
        self.disabled_solver: list[type[Solver] | type[EigSolver]] = []

        self.symmetry_limit: float = float(_SYMMETRY_LIMIT)
        self.force_symmetric: bool = False
        self.use_sorter: bool = False
        self.use_preconditioner: bool = False
        self.use_direct: bool = True

        self.save_disk: bool = False

        for solverkey, solver in self.solvers.items():
            if solver.name in _FORCED_SOLVER:
                self.forced_solver.append(solver)

    def _set_name(self, thread_nr: int, proc_nr: int):
        self.pre = f"p{int(proc_nr):02d}/t{int(thread_nr):02d}"

    def _set_environ_variables(self) -> None:
        os.environ["EMERGE_MP_SOLVER"] = "_".join(
            [solver.name for solver in self.forced_solver]
        )
        os.environ["EMERGE_SYM_LIMIT"] = str(self.symmetry_limit)

    def __str__(self) -> str:
        return "SolveRoutine()"

    def _legal_solver(self, solver: Solver | EigSolver) -> bool:
        """Checks if a solver is a legal option.

        Args:
            solver (Solver): The solver to test against

        Returns:
            bool: If the solver is legal
        """
        if any(
            isinstance(solver, solvertype.__class__)
            for solvertype in self.disabled_solver
        ):
            logger.warning(
                self.pre
                + f"The selected solver {solver} cannot be used as it is disabled."
            )
            return False
        if self.parallel == "MT" and not solver.stype == SolverType.SINGLE_MT:
            logger.warning(
                self.pre
                + f"The selected solver {solver} cannot be used in MultiThreading as it does not release the GIL"
            )
            return False
        return True

    @property
    def all_solvers(self) -> list[Solver]:
        return list(
            [
                solver
                for solver in self.solvers.values()
                if not isinstance(solver, EigSolver)
            ]
        )

    @property
    def all_eig_solvers(self) -> list[EigSolver]:
        return list(
            [
                solver
                for solver in self.solvers.values()
                if isinstance(solver, EigSolver)
            ]
        )

    def _try_solver(self, solver_type: EMSolver) -> Solver:
        """Try to use the selected solver or else find another one that is working.

        Args:
            solver_type (EMSolver): The solver type to try

        Raises:
            RuntimeError: Error if no valid solver is found.

        Returns:
            Solver: The working solver.
        """
        solver = self.solvers[solver_type]
        if self._legal_solver(solver):
            return solver  # type: ignore
        for alternative in self.all_solvers:
            if self._legal_solver(alternative):
                logger.debug(self.pre + f"Falling back on legal solver: {alternative}")
                return alternative
        raise RuntimeError(
            self.pre
            + f"No legal solver could be found. The following are disabled: {self.disabled_solver}"
        )

    def duplicate(self) -> SolveRoutine:
        """Creates a copy of this SolveRoutine class object.

        Returns:
            SolveRoutine: The copied version
        """
        new_routine = self.__class__()
        new_routine.parallel = self.parallel
        new_routine.smart_search = self.smart_search
        new_routine.forced_solver = self.forced_solver
        new_routine.force_symmetric = self.force_symmetric
        for tpe, solver in self.solvers.items():
            new_routine.solvers[tpe] = solver.duplicate()
        return new_routine

    def set_solver(self, *solvers: EMSolver | EigSolver | Solver) -> None:
        """Set a given Solver class instance as the main solver.
        Solvers will be checked on validity for the given problem.

        Args:
            solver (EMSolver | Solver): The solver objects
        """

        for solver in solvers:
            if isinstance(solver, EMSolver):
                self.forced_solver = [self.solvers[solver]]
            else:
                self.forced_solver = [solver]

    def disable(self, *solvers: EMSolver) -> None:
        """Disable a given Solver class instance as the main solver.
        Solvers will be checked on validity for the given problem.

        Args:
            solver (EMSolver): The solver objects
        """
        for solver in solvers:
            if isinstance(solver, EMSolver):
                self.disabled_solver.append(self.solvers[solver])
            else:
                self.disabled_solver.append(solver)

    def _configure_routine(
        self,
        parallel: Literal["SI", "MT", "MP"] = "SI",
        smart_search: bool = False,
        thread_nr: int = 0,
        proc_nr: int = 0,
    ) -> SolveRoutine:
        """Configure the solver with the given settings

        Args:
            parallel (Literal['SI','MT','MP'], optional):
                The solver parallism, Defaults to 'SI'.
                    - "SI" = Single threaded
                    - "MT" = Multi threaded
                    - "MP" = Multi-processing,
            smart_search (bool, optional): Wether to use smart-search solvers
            for eigenmode problems. Defaults to False.

        Returns:
            SolveRoutine: The same SolveRoutine object.
        """
        self.parallel = parallel
        self.smart_search = smart_search
        if thread_nr != 1 or proc_nr != 1:
            self._set_name(thread_nr, proc_nr)
            for solver in self.solvers.values():
                if not isinstance(solver, (Solver, EigSolver)):
                    continue
                solver.pre = self.pre
        return self

    def configure(self, pivoting_threshold: float | None = None) -> None:
        """Sets general user configurations for all solvers.

        Args:
            pivoting_threshold (float | None, optional):
                The diagonal pivoting threshold used in direct solvers. Standard values are 0.001.
                In simulations with a very low surface impedance (such as with copper walls) a much
                lower pivoting threshold is desired.
        """
        for solver in self.solvers.values():
            if isinstance(solver, Solver):
                solver.set_options(pivoting_threshold=pivoting_threshold)

    def reset(self, reset_solver_preference: bool = False, hard: bool = False) -> None:
        """Reset all solver states"""
        for solver in self.solvers.values():
            solver.reset()
        self.sorter.reset()
        self.parallel = "SI"
        self.smart_search = False
        self.force_symmetric = False
        if reset_solver_preference:
            self.forced_solver = []
            self.disabled_solver = []
        if hard:
            self.solvers: dict[EMSolver, Solver | EigSolver] = {
                slv: slv.create_solver(self.pre) for slv in EMSolver
            }
            self.solvers = {
                key: solver
                for key, solver in self.solvers.items()
                if solver is not None
            }

    def _get_solver(
        self, A: csc_matrix, b: np.ndarray, matrix_type: MatrixType, direct: bool = True
    ) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: Returns the direct solver

        """
        for solver in self.forced_solver:
            if solver is None:
                continue
            if not self._legal_solver(solver):
                continue
            if isinstance(solver, Solver):
                return solver
        return self.pick_solver(A, b, matrix_type, direct)

    def pick_solver(
        self, A: csc_matrix, b: np.ndarray, matrix_type: MatrixType, direct: bool = True
    ) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: Returns the direct solver

        """
        if direct:
            if matrix_type is MatrixType.SPD and _SKSP_AVAILABLE:
                return self._try_solver(EMSolver.CHOLMOD)
            return self._try_solver(EMSolver.SUPERLU)

        else:
            if matrix_type in (MatrixType.SPD, MatrixType.HPD):
                return self._try_solver(EMSolver.CG)
            else:
                return self._try_solver(EMSolver.BICGSTAB)

    def _get_eig_solver(
        self, A: csc_matrix, b: csc_matrix, direct: bool | None = None
    ) -> EigSolver:
        """Returns the relevant eigenmode Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for
            direct (bool): If the direct solver should be used.

        Returns:
            Solver: Returns the solver object

        """
        for solver in self.forced_solver:
            if isinstance(solver, EigSolver):
                return solver  # type: ignore
        if direct or A.shape[0] < 1000:
            return self.solvers[EMSolver.LAPACK]  # type: ignore
        else:
            return self.solvers[EMSolver.SMART_ARPACK]  # type: ignore

    def _get_eig_solver_bma(
        self, A: csc_matrix, b: csc_matrix, direct: bool | None = None
    ) -> EigSolver:
        """Returns the relevant eigenmode Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.

        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for
            direct (bool): If the direct solver should be used.

        Returns:
            Solver: Returns the solver object

        """
        for solver in self.forced_solver:
            if isinstance(solver, EigSolver):
                return solver

        if direct or A.shape[0] < 500:
            return self.solvers[EMSolver.LAPACK]  # type: ignore
        else:
            return self.solvers[EMSolver.SMART_ARPACK_BMA]  # type: ignore

    def solve(
        self,
        A: csc_matrix | csc_matrix,
        b: np.ndarray,
        solve_ids: np.ndarray,
        direct: bool = True,
        id: int = -1,
        matrix_type: MatrixType = MatrixType.UNKNOWN,
    ) -> tuple[np.ndarray, SolveReport]:
        """Solve the system of equations defined by Ax=b for x.

        Solve is the main function call to solve a linear system of equations defined by Ax=b.
        The solve routine will go through the required steps defined in the routine to tackle the problme.

        Args:
            A (np.ndarray | csc_matrix): The (Sparse) matrix
            b (np.ndarray): The source vector
            solve_ids (np.ndarray): A vector of ids for which to solve the problem. For EM problems this
            implies all non-PEC degrees of freedom.
            reuse (bool): Whether to reuse the existing factorization if it exists.

        Returns:
            np.ndarray: The resultant solution.
        """
        if matrix_type is MatrixType.UNKNOWN:
            matrix_type = classify_matrix(A)

        if self.save_disk:
            name_A = f'SparseMatrix_{id}.npz'
            name_b = f'SolVector_{id}.npy'
            save_npz(name_A, A)
            np.save(name_b, b)
            logger.info(f'Saved {name_A} and {name_b}')
        logger.debug(f"{_pfx(self.pre, id)} matrix type: {matrix_type}")

        if b.ndim == 1:
            b = b.reshape((b.shape[0], 1))
        solver: Solver = self._get_solver(A, b, matrix_type=matrix_type, direct=direct)

        solver.initialize()
        solver.set_symmetry(matrix_type.symmetric or self.force_symmetric)

        NF = A.shape[0]
        NS = solve_ids.shape[0]
        NB = b.shape[1]

        Asel = A[:, solve_ids][solve_ids, :]
        bsel = b[solve_ids, :]
        nnz = Asel.nnz

        logger.debug(
            f"{_pfx(self.pre, id)} Removed {NF - NS} prescribed DOFs ({NS:,} left, {nnz:,}≠0)"
        )

        if solver.real_only:
            logger.debug(f"{_pfx(self.pre, id)} Converting to real matrix")
            Asel, bsel = complex_to_real_block(Asel, bsel)

        # SORT
        sorter = "None"
        if solver.req_sorter and self.use_sorter:
            sorter = str(self.sorter)
            Asorted, bsorted = self.sorter.sort(Asel, bsel)
        else:
            Asorted, bsorted = Asel, bsel

        start = time.time()

        # Preconditioner
        precon = "None"
        if self.use_preconditioner:
            if not solver.own_preconditioner:
                self.precon.init(Asorted, bsorted)
                precon = str(self.precon)

        x_solved, report = solver.solve(Asorted, bsorted, self.precon, id=id)

        end = time.time()
        simtime = end - start
        logger.info(f"{_pfx(self.pre, id)} Elapsed time taken: {simtime:.3f} seconds")
        logger.debug(
            f"{_pfx(self.pre, id)} O(N^2) performance = {(NS ** (2)) / ((end - start + 1e-6) * 1e6):.1f} MDoF/s"
        )

        if self.use_sorter and solver.req_sorter:
            x = self.sorter.unsort(x_solved)
        else:
            x = x_solved

        if solver.real_only:
            logger.debug(f"{_pfx(self.pre, id)} Converting back to complex matrix")
            x = real_to_complex_block(x)

        solution = np.zeros((NF, NB), dtype=A.dtype)

        solution[solve_ids, :] = x

        logger.debug(f"{_pfx(self.pre, id)} Solver complete!")
        report.jobid = id
        report.sorter = str(sorter)
        report.simtime = simtime
        report.nnz = A.nnz
        report.ndof = b.shape[0]
        report.nnz_solve = Asorted.nnz
        report.ndof_solve = bsorted.shape[0]
        report.precon = precon
        report.worker_name = self.pre

        return solution, report

    def eig_boundary(
        self,
        A: csc_matrix,
        B: np.ndarray,
        solve_ids: np.ndarray,
        nmodes: int = 6,
        direct: bool | None = None,
        target_k0: float = 0.0,
        which: str = "LM",
        sign: float = -1,
    ) -> tuple[np.ndarray, np.ndarray, SolveReport]:
        """Find the eigenmodes for the system Ax = λBx for a boundary mode problem

        For generalized eigenvalue problems of boundary mode analysis studies, the equation is: Ae = -β²Be

        Args:
            A (csc_matrix): The Stiffness matrix
            B (csc_matrix): The mass matrix
            solve_ids (np.ndarray): The free nodes (non PEC)
            nmodes (int): The number of modes to solve for. Defaults to 6
            direct (bool): If the direct solver should be used (always). Defaults to False
            target_k0 (float): The k0 value to search around
            which (str): The search method. Defaults to 'LM' (Largest Magnitude)
            sign (float): The sign of the eigenvalue expression. Defaults to -1

        Returns:
            np.ndarray: The eigen values
            np.ndarray: The eigen vectors
            SolveReport: The solution report
        """
        solver = self._get_eig_solver_bma(A, B, direct=direct)
        solver.initialize()
        NF = A.shape[0]
        NS = solve_ids.shape[0]

        logger.debug(self.pre + f" Removing {NF - NS} prescribed DOFs ({NS} left)")

        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]

        start = time.time()
        eigen_values, eigen_modes = solver.eig(
            Asel, Bsel, nmodes, target_k0, which, sign=sign
        )
        end = time.time()

        simtime = end - start
        return (
            eigen_values,
            eigen_modes,
            SolveReport(
                ndof=A.shape[0],
                nnz=A.nnz,
                ndof_solve=Asel.shape[0],
                nnz_solve=Asel.nnz,
                simtime=simtime,
                solver=str(solver),
                sorter="None",
                precon="None",
            ),
        )

    def eig(
        self,
        A: csc_matrix,
        B: np.ndarray,
        solve_ids: np.ndarray,
        nmodes: int = 6,
        direct: bool | None = None,
        target_f0: float = 0.0,
        which: str = "LM",
    ) -> tuple[np.ndarray, np.ndarray, SolveReport]:
        """
        Find the eigenmodes for the system Ax = λBx for a boundary mode problem
        
        Args:
            A (csr_matrix): The Stiffness matrix
            B (csr_matrix): The mass matrix
            solve_ids (np.ndarray): The free nodes (non PEC)
            nmodes (int): The number of modes to solve for. Defaults to 6
            direct (bool): If the direct solver should be used (always). Defaults to False
            target_k0 (float): The k0 value to search around
            which (str): The search method. Defaults to 'LM' (Largest Magnitude)
            sign (float): The sign of the eigenvalue expression. Defaults to -1\
        Returns:
            np.ndarray: The resultant solution.
        """
        solver = self._get_eig_solver(A, B, direct=direct)

        NF = A.shape[0]
        NS = solve_ids.shape[0]

        logger.debug(self.pre + f" Removing {NF - NS} prescribed DOFs ({NS} left)")

        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]

        start = time.time()
        eigen_values, eigen_modes = solver.eig(
            Asel, Bsel, nmodes, target_f0, which, sign=1.0
        )
        end = time.time()
        simtime = end - start

        Nsols = eigen_modes.shape[1]
        sols = np.zeros((NF, Nsols), dtype=np.complex128)
        for i in range(Nsols):
            sols[solve_ids, i] = eigen_modes[:, i]

        return (
            eigen_values,
            sols,
            SolveReport(
                ndof=A.shape[0],
                nnz=A.nnz,
                ndof_solve=Asel.shape[0],
                nnz_solve=Asel.nnz,
                simtime=simtime,
                solver=str(solver),
                sorter="None",
                precon="None",
            ),
        )

class AutomaticRoutine(SolveRoutine):
    """Defines the Automatic Routine for EMerge as a Singleton."""
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        # Prevent re-initialization if the singleton instance already exists
        if getattr(self, "_initialized", False):
            return
        
        super().__init__(*args, **kwargs)
        self._initialized = True

    def pick_solver(
        self, A: np.ndarray, b: np.ndarray, matrix_type: MatrixType, direct: bool = True
    ) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        The current implementation only looks at matrix size to select the best solver. Matrices
        with a large size will use iterative solvers while smaller sizes will use either Pardiso
        for medium sized problems or SPSolve for small ones.

        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for
            matrix_type (MatrixType): Type of the matrix (e.g., SPD, HPD)
            direct (bool): Whether to use a direct solver

        Returns:
            Solver: A solver object appropriate for solving the problem.

        """

        N = A.shape[0]
        if not direct:
            if matrix_type in (MatrixType.SPD, MatrixType.HPD):
                return self._try_solver(EMSolver.CG)
            else:
                return self._try_solver(EMSolver.BICGSTAB)

        if N < 10_000:
            if matrix_type is MatrixType.SPD and _SKSP_AVAILABLE:
                return self._try_solver(EMSolver.CHOLMOD)
            return self._try_solver(EMSolver.SUPERLU)
        if self.parallel == "SI":
            if matrix_type is MatrixType.SPD and _SKSP_AVAILABLE:
                return self._try_solver(EMSolver.CHOLMOD)
            if _PARDISO_AVAILABLE:
                return self._try_solver(EMSolver.PARDISO)
            elif _AASDS_AVAILABLE:
                return self._try_solver(EMSolver.AASDS)
            elif _MUMPS_AVAILABLE:
                return self._try_solver(EMSolver.MUMPS)
            elif _UMFPACK_AVAILABLE:
                return self._try_solver(EMSolver.UMFPACK)
            else:
                return self._try_solver(EMSolver.SUPERLU)
        elif self.parallel == "MP":
            if _AASDS_AVAILABLE:
                return self._try_solver(EMSolver.AASDS)
            if _UMFPACK_AVAILABLE:
                return self._try_solver(EMSolver.UMFPACK)
            else:
                return self._try_solver(EMSolver.SUPERLU)
        elif self.parallel == "MT":
            return self._try_solver(EMSolver.SUPERLU)

        return self._try_solver(EMSolver.SUPERLU)