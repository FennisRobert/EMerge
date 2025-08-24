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


from __future__ import annotations
from scipy.sparse import csr_matrix # type: ignore
from scipy.sparse.csgraph import reverse_cuthill_mckee # type: ignore
from scipy.sparse.linalg import bicgstab, gmres, gcrotmk, eigs, splu # type: ignore
from scipy.linalg import eig # type: ignore
from scipy import sparse # type: ignore
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
import platform
import time
from typing import Literal, Callable
from enum import Enum

_PARDISO_AVAILABLE = False
_UMFPACK_AVAILABLE = False
_CUDSS_AVAILABLE = False

""" Check if the PC runs on a non-ARM architechture
If so, attempt to import PyPardiso (if its installed)
"""


############################################################
#                          PARDISO                         #
############################################################

if 'arm' not in platform.processor():
    from .solve_interfaces.pardiso_interface import PardisoInterface
    _PARDISO_AVAILABLE = True


############################################################
#                          UMFPACK                         #
############################################################


try:
    import scikits.umfpack as um # type: ignore
    _UMFPACK_AVAILABLE = True
except ModuleNotFoundError:
    logger.debug('UMFPACK not found, defaulting to SuperLU')

############################################################
#                           CUDSS                          #
############################################################

try:
    from .solve_interfaces.cudss_interface import CuDSSInterface
    _CUDSS_AVAILABLE = True
except ModuleNotFoundError:
    pass
except ImportError as e:
    logger.error('Error while importing CuDSS dependencies:')
    logger.exception(e)
    
############################################################
#                       SOLVE REPORT                       #
############################################################

@dataclass
class SolveReport:
    simtime: float = -1.0
    jobid: int = -1
    ndof: int = -1
    nnz: int = -1
    ndof_solve: int = -1
    nnz_solve: int = -1
    exit_code: int = 0
    solver: str = 'None'
    sorter: str = 'None'
    precon: str = 'None'
    aux: dict[str, str] = field(default_factory=dict)

    def add(self, **kwargs: str):
        for key, value in kwargs.items():
            self.aux[key] = str(value)
    
    def pretty_print(self, print_cal: Callable | None = None):
        if print_cal is None:
            print_cal = print
        # Set column widths
        col1_width = 22  # Wider key column
        col2_width = 40  # Value column
        total_width = col1_width + col2_width + 5  # +5 for borders/padding

        def row(key, val):
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print_cal(f"| {key:<{col1_width}} | {val_str:<{col2_width}} |") # ty: ignore

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
        print_cal(border)

        if self.aux:
            print_cal(f"| {'Additional Info':^{total_width - 2}} |")
            print_cal(border)
            for k, v in self.aux.items():
                row(str(k), v)
            print_cal(border)

############################################################
#                 EIGENMODE FILTER ROUTINE                #
############################################################

def filter_real_modes(eigvals: np.ndarray, eigvecs: np.ndarray, 
                      k0: float, ermax: complex, urmax: complex, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Given arrays of eigenvalues `eigvals` and eigenvectors `eigvecs` (cols of shape (N,)),
    and a free‐space wavenumber k0, return only those eigenpairs whose eigenvalue can
    correspond to a real propagation constant β (i.e. 0 ≤ β² ≤ k0²·ermax·urmax).

    Assumes that `ermax` and `urmax` are defined in the surrounding scope.

    Parameters
    ----------
    eigvals : 1D array_like of float
        The generalized eigenvalues (β² candidates).
    eigvecs : 2D array_like, shape (N, M)
        The corresponding eigenvectors, one column per eigenvalue.
    k0 : float
        Free‐space wavenumber.

    Returns
    -------
    filtered_vals : 1D ndarray
        Subset of `eigvals` satisfying 0 ≤ eigval ≤ k0²·ermax·urmax (within numerical tol).
    filtered_vecs : 2D ndarray
        Columns of `eigvecs` corresponding to `filtered_vals`.
    """
    minimum = 1
    extremum = (k0**2) * ermax * urmax * 2
    
    mask = (sign*eigvals <= extremum) & (sign*eigvals >= minimum)
    filtered_vals = eigvals[mask]
    filtered_vecs = eigvecs[:, mask]
    k0vals = np.sqrt(sign*filtered_vals)
    order = np.argsort(np.abs(k0vals))# ascending distance
    filtered_vals = filtered_vals[order]             # reorder eigenvalues
    filtered_vecs = filtered_vecs[:, order] 
    return filtered_vals, filtered_vecs


############################################################
#               EIGENMODE ORTHOGONALITY CHECK              #
############################################################

def filter_unique_eigenpairs(eigen_values: list[complex], 
                             eigen_vectors: list[np.ndarray], tol=-3) -> tuple[list[complex], list[np.ndarray]]:
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
    selected: list = []
    indices: list = []
    for i in range(len(eigen_vectors)):
        
        vec = eigen_vectors[i]
        vec = vec / np.linalg.norm(vec)  # Normalize

        # Check orthogonality against selected vectors
        if all(10*np.log10(abs(np.dot(vec, sel))) < tol for sel in selected):
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
    A_r = sparse.csr_matrix(A.real)
    A_i = sparse.csr_matrix(A.imag)
    #  [ ReA  -ImA ]
    #  [ ImA   ReA ]
    upper = sparse.hstack([A_r, -A_i])
    lower = sparse.hstack([A_i,  A_r])
    A_hat = sparse.vstack([upper, lower]).tocsr()

    b_hat = np.hstack([b.real, b.imag])
    return A_hat, b_hat

def real_to_complex_block(x):
    """Return x = (x_r, x_i) as complex vector."""
    n = x.shape[0] // 2
    x_r = x[:n]
    x_i = x[n:]
    return x_r + 1j * x_i


############################################################
#                  BASE CLASS DEFINITIONS                 #
############################################################

class SimulationError(Exception):
    pass

class Sorter:
    """ A Generic class that executes a sort on the indices.
    It must implement a sort and unsort method.
    """
    def __init__(self):
        self.perm = None
        self.inv_perm = None

    def reset(self):
        """ Reset the permuation vectors."""
        self.perm = None
        self.inv_perm = None

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def sort(self, A: csr_matrix, b: np.ndarray, reuse_sorting: bool = False) -> tuple[csr_matrix, np.ndarray]:
        return A,b
    
    def unsort(self, x: np.ndarray) -> np.ndarray:
        return x

class Preconditioner:
    """A Generic class defining a preconditioner as attribute .M based on the
    matrix A and b. This must be generated in the .init(A,b) method.
    """
    def __init__(self):
        self.M: np.ndarray = None

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def init(self, A: csr_matrix, b: np.ndarray) -> None:
        raise NotImplementedError('')

class Solver:
    """ A generic class representing a solver for the problem Ax=b
    
    A solver class has two class attributes.
     - real_only: defines if the solver can only deal with real numbers. In this case
    the solve routine will automatically provide A and b in real number format.
     - req_sorter: defines if this solver requires the use of a sorter algorithm. By setting
     it to False, the SolveRoutine will not use the default sorting algorithm.
    """
    real_only: bool = False
    req_sorter: bool = False

    def __init__(self):
        self.own_preconditioner: bool = False

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def duplicate(self) -> Solver:
        return self.__class__()

    def set_options(self, pivoting_threshold: float | None = None) -> None:
        """Write generic simulation options to the solver object. 
        Options may be ignored depending on the type of solver used."""
        pass
        
    def solve(self, A: csr_matrix, b: np.ndarray, precon: Preconditioner, 
              reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        raise NotImplementedError("This classes Ax=B solver method is not implemented.")
    
    def reset(self) -> None:
        """Reset state variables like numeric and symbollic factorizations."""
        pass

class EigSolver:
    """ A generic class representing a solver for the eigenvalue problem Ax=λBx
    
    A solver class has two class attributes.
     - real_only: defines if the solver can only deal with real numbers. In this case
    the solve routine will automatically provide A and b in real number format.
     - req_sorter: defines if this solver requires the use of a sorter algorithm. By setting
     it to False, the SolveRoutine will not use the default sorting algorithm.
    """
    real_only: bool = False
    req_sorter: bool = False

    def __init__(self):
        self.own_preconditioner: bool = False

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def duplicate(self) -> Solver:
        return self.__class__()

    def eig(self, A: csr_matrix | csr_matrix, B: csr_matrix | csr_matrix, nmodes: int = 6, 
            target_k0: float = 0.0, which: str = 'LM', sign: float = 1.):
        raise NotImplementedError("This classes eigenmdoe solver method is not implemented.")
    
    def reset(self) -> None:
        """Reset state variables like numeric and symbollic factorizations."""
        pass


############################################################
#                          SORTERS                         #
############################################################


class ReverseCuthillMckee(Sorter):
    """ Implements the Reverse Cuthill-Mckee sorting."""
    def __init__(self):
        super().__init__()
        

    def sort(self, A, b, reuse_sorting: bool = False):
        if not reuse_sorting:
            logger.debug('Generating Reverse Cuthill-Mckee sorting.')
            self.perm = reverse_cuthill_mckee(A)
            self.inv_perm = np.argsort(self.perm)
        logger.debug('Applying Reverse Cuthill-Mckee sorting.')
        Asorted = A[self.perm, :][:, self.perm]
        bsorted = b[self.perm]
        return Asorted, bsorted
    
    def unsort(self, x: np.ndarray):
        logger.debug('Reversing Reverse Cuthill-Mckee sorting.')
        return  x[self.inv_perm]
    

############################################################
#                      PRECONDITIONERS                     #
############################################################


class ILUPrecon(Preconditioner):
    """ Implements the incomplete LU preconditioner on matrix A. """
    def __init__(self):
        super().__init__()
        self.M = None
        self.fill_factor = 10
        self.options: dict[str, str] = dict(SymmetricMode=True)

    def init(self, A, b):
        logger.info("Generating ILU Preconditioner")
        self.ilu = sparse.linalg.spilu(A, drop_tol=1e-2, fill_factor=self.fill_factor, # ty: ignore
                                       permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.001, 
                                       options=self.options) 
        self.M = sparse.linalg.LinearOperator(A.shape, self.ilu.solve) # ty: ignore


############################################################
#                     ITERATIVE SOLVERS                    #
############################################################


class SolverBicgstab(Solver):
    """ Implements the Bi-Conjugate Gradient Stabilized method"""
    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(f'Iteration {convergence:.4f}')

    def solve(self, A, b, precon, reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f'[ID={id}] Calling BiCGStab.')
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = bicgstab(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = bicgstab(A, b, atol=self.atol, callback=self.callback)
        return x, SolveReport(solver=str(self), exit_code=info)

class SolverGCROTMK(Solver):
    """ Implements the GCRO-T(m,k) Iterative solver. """
    def __init__(self):
        super().__init__()
        self.atol = 1e-5
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(f'Iteration {convergence:.4f}')

    def solve(self, A: csr_matrix, b: np.ndarray, precon: Preconditioner, reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f'[ID={id}] Calling GCRO-T(m,k) algorithm')
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gcrotmk(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = gcrotmk(A, b, atol=self.atol, callback=self.callback)
        return x, SolveReport(solver=str(self), exit_code=info)

class SolverGMRES(Solver):
    """ Implements the GMRES solver. """
    real_only = False
    req_sorter = True

    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, norm):
        #convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(f'Iteration {norm:.4f}')

    def solve(self, A, b, precon, reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f'[ID={id}] Calling GMRES Function.')
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gmres(A, b, M=precon.M, atol=self.atol, callback=self.callback, callback_type='pr_norm')
        else:
            x, info = gmres(A, b, atol=self.atol, callback=self.callback, restart=500, callback_type='pr_norm')
        return x, SolveReport(solver=str(self), exit_code=info)


############################################################
#                      DIRECT SOLVERS                     #
############################################################


class SolverSuperLU(Solver):
    """ Implements Scipi's direct SuperLU solver."""
    req_sorter: bool = False
    real_only: bool = False
    
    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self.options: dict[str, str] = dict(SymmetricMode=True, Equil=False, IterRefine='SINGLE')
        self._pivoting_threshold: float = 0.001
        self.lu = None
    
    def duplicate(self) -> Solver:
        new_solver = self.__class__()
        new_solver._pivoting_threshold = self._pivoting_threshold
        return new_solver

    def set_options(self,
                    pivoting_threshold: float | None = None) -> None:
        if pivoting_threshold is not None:
            self._pivoting_threshold = pivoting_threshold
    
    def solve(self, A, b, precon, reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f'[ID={id}] Calling SuperLU Solver.')
        self.single = True
        if not reuse_factorization:
            self.lu = splu(A, permc_spec='MMD_AT_PLUS_A', relax=0, diag_pivot_thresh=self._pivoting_threshold, options=self.options)
        x = self.lu.solve(b)
        aux = {
            "pivoting threshold": str(self._pivoting_threshold)
        }
        return x, SolveReport(solver=str(self), exit_code=0, aux=aux)

class SolverUMFPACK(Solver):
    """ Implements the UMFPACK Sparse SP solver."""
    req_sorter = False
    real_only = False

    def __init__(self):
        super().__init__()
        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self.umfpack: um.UmfpackContext = um.UmfpackContext('zl')
        self.umfpack.control[um.UMFPACK_PRL] = 0 # ty: ignore
        self.umfpack.control[um.UMFPACK_IRSTEP] = 2 # ty: ignore
        self.umfpack.control[um.UMFPACK_STRATEGY] = um.UMFPACK_STRATEGY_SYMMETRIC # ty: ignore
        self.umfpack.control[um.UMFPACK_ORDERING] = 3 # ty: ignore
        self.umfpack.control[um.UMFPACK_PIVOT_TOLERANCE] = 0.001 # ty: ignore
        self.umfpack.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = 0.001 # ty: ignore
        self.umfpack.control[um.UMFPACK_BLOCK_SIZE] = 64 # ty: ignore
        self.umfpack.control[um.UMFPACK_FIXQ] = -1 # ty: ignore
        
        # SETTINGS
        self._pivoting_threshold: float = 0.001

        self.fact_symb: bool = False

    def reset(self) -> None:
        self.fact_symb = False
    
    def set_options(self,
                    pivoting_threshold: float | None = None) -> None:
            if pivoting_threshold is not None:
                self.umfpack.control[um.UMFPACK_PIVOT_TOLERANCE] = pivoting_threshold # ty: ignore
                self.umfpack.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = pivoting_threshold # ty: ignore
                self._pivoting_threshold = pivoting_threshold

    def duplicate(self) -> Solver:
        new_solver = self.__class__()
        new_solver.set_options(pivoting_threshold = self._pivoting_threshold)
        return new_solver

    def solve(self, A, b, precon, reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f'[ID={id}] Calling UMFPACK Solver.')
        A.indptr  = A.indptr.astype(np.int64)
        A.indices = A.indices.astype(np.int64)
        if self.fact_symb is False:
            logger.debug(f'[ID={id}] Executing symbollic factorization.')
            self.umfpack.symbolic(A)
            self.fact_symb = True
        if not reuse_factorization:
            #logger.debug('Executing numeric factorization.')
            self.umfpack.numeric(A)
            self.A = A
        x = self.umfpack.solve(um.UMFPACK_A, self.A, b, autoTranspose = False ) # ty: ignore
        aux = {
            "Pivoting Threshold": str(self._pivoting_threshold),
        }
        return x, SolveReport(solver=str(self), exit_code=0, aux=aux)

class SolverPardiso(Solver):
    """ Implements the PARDISO solver through PyPardiso. """
    real_only: bool = False
    req_sorter: bool = False

    def __init__(self):
        super().__init__()
        self.solver: PardisoInterface = PardisoInterface()
        self.fact_symb: bool = False
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def solve(self, A, b, precon, reuse_factorization: bool = False, id: int = -1) -> tuple[np.ndarray, SolveReport]:
        logger.info(f'[ID={id}] Calling Pardiso Solver')
        if self.fact_symb is False:
            logger.debug(f'[ID={id}] Executing symbollic factorization.')
            self.solver.symbolic(A)
            self.fact_symb = True
        if not reuse_factorization:
            self.solver.numeric(A)
            self.A = A
        x, error = self.solver.solve(A, b)
        if error != 0:
            logger.error(f'[ID={id}] Terminated with error code {error}')
            logger.error(self.solver.get_error(error))
            raise SimulationError(f'[ID={id}] PARDISO Terminated with error code {error}')
        aux = {}
        return x, SolveReport(solver=str(self), exit_code=error, aux=aux)
    

class CuDSSSolver(Solver):
    real_only = False
    def __init__(self):
        self._cudss = CuDSSInterface()
        self.fact_symb: bool = False
        self.fact_numb: bool = False
        self._cudss._PRES = 2

    def reset(self) -> None:
        self.fact_symb = False
        self.fact_numb = False

    def solve(self, A, b, precon, reuse_factorization: bool = False, id: int = -1):
        logger.info(f'[ID={id}] Calling cuDSS Solver')
        
        if self.fact_symb is False:
            logger.debug('Executing symbollic factorization')
            x = self._cudss.from_symbolic(A,b)
            self.fact_symb = True
        else:
            if reuse_factorization:
                x = self._cudss.from_solve(b)
            else:
                x = self._cudss.from_numeric(A,b)
        
        return x, SolveReport(solver=str(self), exit_code=0, aux={})


############################################################
#                 DIRECT EIGENMODE SOLVERS                #
############################################################

class SolverLAPACK(EigSolver):

    def __init__(self):
        super().__init__()
    
    def eig(self, 
            A: csr_matrix | csr_matrix, 
            B: csr_matrix | csr_matrix,
            nmodes: int = 6,
            target_k0: float = 0,
            which: str = 'LM',
            sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
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
        logger.debug('Calling LAPACK eig solver')
        lam, vecs = eig(A.toarray(), B.toarray(), overwrite_a=True, overwrite_b=True, check_finite=False)
        lam, vecs = filter_real_modes(lam, vecs, target_k0, 2, 2, sign=sign)
        return lam, vecs
    

############################################################
#                  ITERATIVE EIGEN SOLVERS                 #
############################################################


class SolverARPACK(EigSolver):
    """ Implements the Scipy ARPACK iterative eigenmode solver."""
    def __init__(self):
        super().__init__()

    def eig(self, 
            A: csr_matrix | csr_matrix, 
            B: csr_matrix | csr_matrix,
            nmodes: int = 6,
            target_k0: float = 0,
            which: str = 'LM',
            sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f'Searching around 	β = {target_k0:.2f} rad/m with ARPACK')
        sigma = sign*(target_k0**2)
        eigen_values, eigen_modes = eigs(A, k=nmodes, M=B, sigma=sigma, which=which)
        return eigen_values, eigen_modes

class SmartARPACK_BMA(EigSolver):
    """ Implements the Scipy ARPACK iterative eigenmode solver with automatic search.
    
    The Solver searches in a geometric range around the target wave constant.
    """
    def __init__(self):
        super().__init__()
        self.symmetric_steps: int = 41
        self.search_range: float = 2.0
        self.energy_limit: float = 1e-4

    def eig(self, 
            A: csr_matrix | csr_matrix, 
            B: csr_matrix | csr_matrix,
            nmodes: int = 6,
            target_k0: float = 0,
            which: str = 'LM',
            sign: float = 1.) -> tuple[np.ndarray, np.ndarray]:

        logger.info(f'Searching around 	β = {target_k0:.2f} rad/m with SmartARPACK (BMA)')
        qs = np.geomspace(1, self.search_range, self.symmetric_steps)
        tot_eigen_values = []
        tot_eigen_modes = []
        energies = []
        for i, q in enumerate(qs):
            # Above target k0
            sigma = sign*((q*target_k0)**2)
            eigen_values, eigen_modes = eigs(A, k=1, M=B, sigma=sigma, which=which)
            energy = np.mean(np.abs(eigen_modes.flatten())**2)
            if energy > self.energy_limit:
                tot_eigen_values.append(eigen_values[0])
                tot_eigen_modes.append(eigen_modes.flatten())
                energies.append(energy)
            if i!=0:
                # Below target k0
                sigma = sign*((target_k0/q)**2)
                eigen_values, eigen_modes = eigs(A, k=1, M=B, sigma=sigma, which=which)
                energy = np.mean(np.abs(eigen_modes.flatten())**2)
                if energy > self.energy_limit:
                    tot_eigen_values.append(eigen_values[0])
                    tot_eigen_modes.append(eigen_modes.flatten())
                    energies.append(energy)
        
        #Sort solutions on mode energy
        val, mode, energy = zip(*sorted(zip(tot_eigen_values,tot_eigen_modes,energies), key=lambda x: x[2], reverse=True))
        eigen_values = np.array(val[:nmodes])
        eigen_modes = np.array(mode[:nmodes]).T

        return eigen_values, eigen_modes
    
class SmartARPACK(EigSolver):
    """ Implements the Scipy ARPACK iterative eigenmode solver with automatic search.
    
    The Solver searches in a geometric range around the target wave constant.
    """
    def __init__(self):
        super().__init__()
        self.symmetric_steps: int = 3
        self.search_range: float = 2.0
        self.energy_limit: float = 1e-4

    def eig(self, 
            A: csr_matrix | csr_matrix, 
            B: csr_matrix | csr_matrix,
            nmodes: int = 6,
            target_k0: float = 0,
            which: str = 'LM',
            sign: float = 1.) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f'Searching around 	β = {target_k0:.2f} rad/m with SmartARPACK')
        qs = np.geomspace(1, self.search_range, self.symmetric_steps)
        tot_eigen_values = []
        tot_eigen_modes = []
        for i, q in enumerate(qs):
            # Above target k0
            sigma = sign*((q*target_k0)**2)
            eigen_values, eigen_modes = eigs(A, k=6, M=B, sigma=sigma, which=which)
            for j in range(eigen_values.shape[0]):
                if eigen_values[j]<(sigma/self.search_range):
                    continue
                tot_eigen_values.append(eigen_values[j])
                tot_eigen_modes.append(eigen_modes[:,j])
            if i!=0:
                # Below target k0
                sigma = sign*((target_k0/q)**2)
                eigen_values, eigen_modes = eigs(A, k=6, M=B, sigma=sigma, which=which)
                for j in range(eigen_values.shape[0]):
                    if eigen_values[j]<(sigma/self.search_range):
                        continue
                    tot_eigen_values.append(eigen_values[j])
                    tot_eigen_modes.append(eigen_modes[:,j])
            tot_eigen_values, tot_eigen_modes = filter_unique_eigenpairs(tot_eigen_values, tot_eigen_modes)
            if len(tot_eigen_values)>nmodes:
                break
        #Sort solutions on mode energy
        val, mode = filter_unique_eigenpairs(tot_eigen_values, tot_eigen_modes)
        val, mode = zip(*sorted(zip(val,mode), key=lambda x: x[0], reverse=False)) # type: ignore
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

    def get_solver(self) -> Solver | EigSolver | None:
        if self==EMSolver.SUPERLU:
            return SolverSuperLU()
        elif self==EMSolver.UMFPACK:
            if _UMFPACK_AVAILABLE is False:
                return None
            else:
                return SolverUMFPACK()
        elif self==EMSolver.PARDISO:
            if _PARDISO_AVAILABLE is False:
                return None
            else:
                return SolverPardiso()
        elif self==EMSolver.LAPACK:
            return SolverLAPACK()
        elif self==EMSolver.ARPACK:
            return SolverARPACK()
        elif self==EMSolver.SMART_ARPACK:
            return SmartARPACK()
        elif self==EMSolver.SMART_ARPACK_BMA:
            return SmartARPACK_BMA()
        elif self==EMSolver.CUDSS:
            if _CUDSS_AVAILABLE is False:
                return None
            else:
                return CuDSSSolver()
        raise ValueError(f'An unsupported Enum case has been reached: {self}')

############################################################
#                       SOLVE ROUTINE                      #
############################################################


class SolveRoutine:
    """ A generic class describing a solve routine.
    A solve routine contains all the relevant sorter preconditioner and solver objects
    and goes through a sequence of steps to solve a linear system or find eigenmodes.

    """
    def __init__(self):
        
        self.sorter: Sorter = ReverseCuthillMckee()
        self.precon: Preconditioner = ILUPrecon()
        self.solvers: dict[EMSolver, Solver | EigSolver] = {slv: slv.get_solver() for slv in EMSolver}
        self.solvers = {key: solver for key, solver in self.solvers.items() if solver is not None}

        self.parallel: Literal['SI','MT','MP'] = 'SI'
        self.smart_search: bool = False
        self.forced_solver: list[Solver | EigSolver] = []
        self.disabled_solver: list[type[Solver]|type[EigSolver]] = []

        self.use_sorter: bool = False
        self.use_preconditioner: bool = False
        self.use_direct: bool = True

    
    def __str__(self) -> str:
        return 'SolveRoutine()'
    
    def _legal_solver(self, solver: Solver | EigSolver) -> bool:
        """Checks if a solver is a legal option.

        Args:
            solver (Solver): The solver to test against

        Returns:
            bool: If the solver is legal
        """
        if any(isinstance(solver, solvertype) for solvertype in self.disabled_solver):
            return False
        return True
    
    @property
    def all_solvers(self) -> list[Solver]:
        return list([solver for solver in self.solvers.values() if not isinstance(solver, EigSolver)])
    
    @property
    def all_eig_solvers(self) -> list[EigSolver]:
        return list([solver for solver in self.solvers.values() if isinstance(solver, EigSolver)])
    

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
                logger.debug(f'Falling back on legal solver: {alternative}')
                return alternative
        raise RuntimeError(f'No legal solver could be found. The following are disabled: {self.disabled_solver}')
    
    def duplicate(self) -> SolveRoutine:
        """Creates a copy of this SolveRoutine class object.

        Returns:
            SolveRoutine: The copied version
        """
        new_routine = self.__class__()
        new_routine.parallel = self.parallel
        new_routine.smart_search = self.smart_search
        new_routine.forced_solver = self.forced_solver
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
                self.forced_solver.append(solver.get_solver())
            else:
                self.forced_solver.append(solver)
    
    def disable(self, *solvers: EMSolver) -> None:
        """Disable a given Solver class instance as the main solver. 
        Solvers will be checked on validity for the given problem.

        Args:
            solver (EMSolver): The solver objects
        """
        for solver in solvers:
            if isinstance(solver, EMSolver):
                self.disabled_solver.append(solver.get_solver().__class__)
            else:
                self.disabled_solver.append(solver.__class__)

    def _configure_routine(self, 
                  parallel: Literal['SI','MT','MP'] = 'SI', 
                  smart_search: bool = False) -> SolveRoutine:
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
        return self

    def configure(self,
                    pivoting_threshold: float | None = None) -> None:
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
        
    def reset(self) -> None:
        """Reset all solver states"""
        for solver in self.solvers.values():
            solver.reset()
        self.sorter.reset()
        self.parallel = 'SI'
        self.smart_search = False
        self.forced_solver = []
        self.disabled_solver = []

    def _get_solver(self, A: csr_matrix, b: np.ndarray) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.
        
        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: Returns the direct solver

        """
        for solver in self.forced_solver:
            if not self._legal_solver(solver):
                continue
            if isinstance(solver, Solver):
                return solver
        return self.pick_solver(A,b)
        
    def pick_solver(self, A: csr_matrix, b: np.ndarray) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.
        
        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: Returns the direct solver

        """
        return self._try_solver(EMSolver.SUPERLU)
    
    def _get_eig_solver(self, A: csr_matrix, b: csr_matrix, direct: bool | None = None) -> EigSolver:
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
            return self.solvers[EMSolver.LAPACK] # type: ignore
        else:
            return self.solvers[EMSolver.SMART_ARPACK] # type: ignore
            
    def _get_eig_solver_bma(self, A: csr_matrix, b: csr_matrix, direct: bool | None = None) -> EigSolver:
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
        
        if direct or A.shape[0] < 1000:
            return self.solvers[EMSolver.LAPACK]  # type: ignore
        else:
            return self.solvers[EMSolver.SMART_ARPACK_BMA]  # type: ignore
    
    def solve(self, A: csr_matrix | csr_matrix, 
              b: np.ndarray, 
              solve_ids: np.ndarray,
              reuse: bool = False,
              id: int = -1) -> tuple[np.ndarray, SolveReport]:
        """ Solve the system of equations defined by Ax=b for x.

        Solve is the main function call to solve a linear system of equations defined by Ax=b.
        The solve routine will go through the required steps defined in the routine to tackle the problme.

        Args:
            A (np.ndarray | csr_matrix | csr_matrix): The (Sparse) matrix
            b (np.ndarray): The source vector
            solve_ids (np.ndarray): A vector of ids for which to solve the problem. For EM problems this
            implies all non-PEC degrees of freedom.
            reuse (bool): Whether to reuse the existing factorization if it exists.

        Returns:
            np.ndarray: The resultant solution.
        """
        solver: Solver = self._get_solver(A, b)

        NF = A.shape[0]
        NS = solve_ids.shape[0]

        A = A.tocsc()
        
        Asel = A[np.ix_(solve_ids, solve_ids)]
        bsel = b[solve_ids]
        nnz = Asel.nnz

        logger.debug(f'[ID={id}] Removed {NF-NS} prescribed DOFs ({NS:,} left, {nnz:,}≠0)')

        if solver.real_only:
            logger.debug(f'[ID={id}] Converting to real matrix')
            Asel, bsel = complex_to_real_block(Asel, bsel)

        # SORT
        sorter = 'None'
        if solver.req_sorter and self.use_sorter:
            sorter = str(self.sorter)
            Asorted, bsorted = self.sorter.sort(Asel, bsel, reuse_sorting=reuse)
        else:
            Asorted, bsorted = Asel, bsel
        
        # Preconditioner
        precon = 'None'
        if self.use_preconditioner:
            if not solver.own_preconditioner:
                self.precon.init(Asorted, bsorted)
                precon = str(self.precon)

        start = time.time()
        
        x_solved, report = solver.solve(Asorted, bsorted, self.precon, reuse_factorization=reuse, id=id)
        end = time.time()
        simtime = end-start
        logger.info(f'[ID={id}] Elapsed time taken: {simtime:.3f} seconds')
        logger.debug(f'[ID={id}] O(N²) performance = {(NS**2)/((end-start+1e-6)*1e6):.3f} MDoF/s')
        
        if self.use_sorter and solver.req_sorter:
            x = self.sorter.unsort(x_solved)
        else:
            x = x_solved
        
        if solver.real_only:
            logger.debug(f'[ID={id}] Converting back to complex matrix')
            x = real_to_complex_block(x)

        solution = np.zeros((NF,), dtype=np.complex128)
        
        solution[solve_ids] = x

        logger.debug(f'[ID={id}] Solver complete!')
        report.jobid = id
        report.sorter = str(sorter)
        report.simtime = simtime
        report.nnz = A.nnz
        report.ndof = b.shape[0]
        report.nnz_solve = Asorted.nnz
        report.ndof_solve = bsorted.shape[0]
        report.precon = precon

        return solution, report
    
    def eig_boundary(self, 
            A: csr_matrix | csr_matrix, 
            B: np.ndarray, 
            solve_ids: np.ndarray,
            nmodes: int = 6,
            direct: bool | None = None,
            target_k0: float = 0.0,
            which: str = 'LM', 
            sign: float=-1) -> tuple[np.ndarray, np.ndarray, SolveReport]:
        """ Find the eigenmodes for the system Ax = λBx for a boundary mode problem

        For generalized eigenvalue problems of boundary mode analysis studies, the equation is: Ae = -β²Be

        Args:
            A (csr_matrix): The Stiffness matrix
            B (csr_matrix): The mass matrix
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

        NF = A.shape[0]
        NS = solve_ids.shape[0]

        logger.debug(f'Removing {NF-NS} prescribed DOFs ({NS} left)')

        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]
        
        start = time.time()
        eigen_values, eigen_modes = solver.eig(Asel, Bsel, nmodes, target_k0, which, sign=sign)
        end = time.time()

        simtime = end-start
        return eigen_values, eigen_modes, SolveReport(ndof=A.shape[0], nnz=A.nnz, 
                                                      ndof_solve=Asel.shape[0], nnz_solve=Asel.nnz, 
                                                      simtime=simtime, solver=str(solver), 
                                                      sorter='None', precon='None')

    def eig(self, 
            A: csr_matrix | csr_matrix, 
            B: np.ndarray, 
            solve_ids: np.ndarray,
            nmodes: int = 6,
            direct: bool | None = None,
            target_f0: float = 0.0,
            which: str = 'LM') -> tuple[np.ndarray, np.ndarray, SolveReport]:
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

        logger.debug(f'Removing {NF-NS} prescribed DOFs ({NS} left)')

        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]
        
        start = time.time()
        eigen_values, eigen_modes = solver.eig(Asel, Bsel, nmodes, target_f0, which, sign=1.0)
        end = time.time()
        simtime = end-start

        Nsols = eigen_modes.shape[1]
        sols = np.zeros((NF, Nsols), dtype=np.complex128)
        for i in range(Nsols):
            sols[solve_ids,i] = eigen_modes[:,i]

        return eigen_values, sols, SolveReport(ndof=A.shape[0], nnz=A.nnz, ndof_solve=Asel.shape[0], nnz_solve=Asel.nnz, simtime=simtime, solver=str(solver), sorter='None', precon='None')

class AutomaticRoutine(SolveRoutine):
    """ Defines the Automatic Routine for EMerge.
    """
        
    def pick_solver(self, A: np.ndarray, b: np.ndarray) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        The current implementation only looks at matrix size to select the best solver. Matrices 
        with a large size will use iterative solvers while smaller sizes will use either Pardiso
        for medium sized problems or SPSolve for small ones.
        
        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: A solver object appropriate for solving the problem.

        """
        N = b.shape[0]
        if N < 10_000:
            return self._try_solver(EMSolver.SUPERLU)
        if self.parallel=='SI':
            if _PARDISO_AVAILABLE:
                return self._try_solver(EMSolver.PARDISO)
            elif _UMFPACK_AVAILABLE:
                return self._try_solver(EMSolver.UMFPACK)
            else:
                return self._try_solver(EMSolver.SUPERLU)
        elif self.parallel=='MP':
            if _UMFPACK_AVAILABLE:
                return self._try_solver(EMSolver.UMFPACK)
            else:
                return self._try_solver(EMSolver.SUPERLU)
        elif self.parallel=='MT':
            return self._try_solver(EMSolver.SUPERLU)
        return self._try_solver(EMSolver.SUPERLU)
    


############################################################
#                    DEFAULT DEFINITION                   #
############################################################

DEFAULT_ROUTINE = AutomaticRoutine()