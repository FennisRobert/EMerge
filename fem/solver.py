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

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import bicgstab, gmres, spsolve, gcrotmk, eigsh, splu, spilu
from scipy.linalg import eig
from scipy import sparse
import numpy as np
from loguru import logger

import platform
import time

if 'arm' not in platform.processor():
    from pypardiso import spsolve as pardiso_solve
#from pyamg.util.utils import get_blocksize

def filter_real_modes(eigvals, eigvecs, k0, ermax, urmax):
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
    tol = -1
    upper_bound = -(k0**2) * ermax * urmax *2

    mask = (eigvals <= tol) & (eigvals >= upper_bound)
    filtered_vals = eigvals[mask]
    filtered_vecs = eigvecs[:, mask]
    k0vals = np.sqrt(-filtered_vals)
    order = np.argsort(np.abs(k0vals - k0))   # ascending distance
    filtered_vals = filtered_vals[order]             # reorder eigenvalues
    filtered_vecs = filtered_vecs[:, order] 
    return filtered_vals, filtered_vecs

def complex_to_real_block(A, b):
    """Return (Â,  b̂) real-augmented representation of A x = b."""
    A_r = sparse.csr_matrix(A.real)
    A_i = sparse.csr_matrix(A.imag)

    #  [ ReA  -ImA ]
    #  [ ImA   ReA ]
    n = A.shape[0]
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

class Sorter:
    
    def __init__(self):
        pass

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def sort(self, A: lil_matrix, b: np.ndarray) -> tuple[lil_matrix, np.ndarray]:
        return A,b
    
    def unsort(self, x: np.ndarray) -> np.ndarray:
        return x

class Preconditioner:
    
    def __init__(self):
        self.M: np.ndarray = None

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def init(self, A: lil_matrix, b: np.ndarray) -> None:
        pass

class Solver:
    
    real_only: bool = False
    req_sorter: bool = False
    def __init__(self):
        self.own_preconditioner: bool = False

    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def solve(self, A: lil_matrix, b: np.ndarray, precon: Preconditioner, reuse_factorization: bool = False) -> tuple[np.ndarray, int]:
        return gmres(A, b, M=precon.M), 0
    
    def eig(self, A: lil_matrix, B: np.ndarray, nmodes: int = 6, target_k0: float = None, which: str = 'LM'):
        raise NotImplementedError("This classes eigs method is not implemented.")
        
## SORTERS

class ReverseCuthillMckee(Sorter):

    def __init__(self):
        super().__init__()
        self.perm = None
        self.inv_perm = None

    def sort(self, A, b):
        logger.info('Generating Reverse Cuthill-Mckee sorting.')
        self.perm = reverse_cuthill_mckee(A)

        self.inv_perm = np.argsort(self.perm)

        Asorted = A[self.perm, :][:, self.perm]
        bsorted = b[self.perm]

        return Asorted, bsorted
    
    def unsort(self, x: np.ndarray):
        logger.info('Reversing Reverse Cuthill-Mckee sorting.')
        return  x[self.inv_perm]
    

## Preconditioners

class ILUPrecon(Preconditioner):

    def __init__(self):
        super().__init__()
        self.M = None
        self.fill_factor = 10

    def init(self, A, b):
        logger.info("Generating ILU Preconditioner")
        self.ilu = sparse.linalg.spilu(A, drop_tol=1e-3, fill_factor=self.fill_factor)
        self.M = sparse.linalg.LinearOperator(A.shape, self.ilu.solve)


## Solvers

class SolverBicgstab(Solver):

    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(f'Iteration {convergence:.4f}')

    def solve(self, A, b, precon):
        logger.info('Calling BiCGStab Function')
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = bicgstab(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = bicgstab(A, b, atol=self.atol, callback=self.callback)
        return x, info

class SolverGCROTMK(Solver):
    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, xk):
        convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(f'Iteration {convergence:.4f}')

    def solve(self, A, b, precon):
        logger.info('Calling GCRO-T(m,k) algorithm')
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gcrotmk(A, b, M=precon.M, atol=self.atol, callback=self.callback)
        else:
            x, info = gcrotmk(A, b, atol=self.atol, callback=self.callback)
        return x, info

class SolverGMRES(Solver):
    real_only = False
    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def callback(self, norm):
        #convergence = np.linalg.norm((self.A @ xk - self.b))
        logger.info(f'Iteration {norm:.4f}')

    def solve(self, A, b, precon):
        logger.info('Calling GMRES Function')
        self.A = A
        self.b = b
        if precon.M is not None:
            x, info = gmres(A, b, M=precon.M, atol=self.atol, callback=self.callback, callback_type='pr_norm')
        else:
            x, info = gmres(A, b, atol=self.atol, callback=self.callback, callback_type='pr_norm')
        return x, info

class SolverSuperLU(Solver):
    req_sorter: bool = True
    real_only: bool = False
    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self._perm_c = None
        self.options: dict[str, str] = dict(SymmetricMode=True)

        self.lu = None
        
    def solve(self, A, b, precon, reuse_factorization: bool = False):
        logger.info('Calling SuperLU Solver')
        if not reuse_factorization:
            self.lu = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.01, options=self.options)
        x = self.lu.solve(b)
        return x, 0

class SolverSP(Solver):
    req_sorter = False
    real_only = False
    def __init__(self):
        super().__init__()
        self.atol = 1e-5

        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def solve(self, A, b, precon):
        logger.info('Calling SP Solver')
        self.A = A
        self.b = b
        x = spsolve(A, b)
        return x, 0
    
class SolverPardiso(Solver):
    real_only: bool = True
    req_sorter: bool = False

    def __init__(self):
        super().__init__()

        self.A: np.ndarray = None
        self.b: np.ndarray = None
    
    def solve(self, A, b, precon, reuse_factorization: bool = False):
        logger.info('Calling Pardiso Solver')
        self.A = A
        self.b = b
        x = pardiso_solve(A, b)
        return x, 0
    
class SolverLAPACK(Solver):

    def __init__(self):
        super().__init__()
    
    def eig(self, 
            A: np.ndarray, 
            B: np.ndarray,
            nmodes: int = 6,
            target_k0: float = 0,
            which: str = 'LM'):
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
        lam, vecs = eig(A.toarray(), B.toarray(), right=True, left=False)
        lam, vecs = filter_real_modes(lam, vecs, target_k0, 2, 2)
        return lam, vecs
    
class SolverARPACK(Solver):

    def __init__(self):
        super().__init__()

    def eig(self, 
            A: np.ndarray, 
            B: np.ndarray,
            nmodes: int = 6,
            target_k0: float = 0,
            which: str = 'LM'):
        logger.info(f'Searching around 	β = {target_k0:.2f} rad/m')
        sigma = -(target_k0**2)
        eigen_values, eigen_modes = eigsh(A, k=nmodes, M=B, sigma=sigma, which=which)
        return eigen_values, eigen_modes

### ROUTINE

class SolveRoutine:

    def __init__(self, 
                 sorter: Sorter, 
                 precon: Preconditioner, 
                 iterative_solver: Solver, 
                 direct_solver: Solver,
                 iterative_eig_solver: Solver,
                 direct_eig_solver: Solver):
        
        self.sorter: Sorter = sorter
        self.precon: Preconditioner = precon

        self.iterative_solver: Solver = iterative_solver
        self.direct_solver: Solver = direct_solver

        self.iterative_eig_solver: Solver = iterative_eig_solver
        self.direct_eig_solver: Solver = direct_eig_solver

        self.use_sorter: bool = True
        self.use_preconditioner: bool = True
        self.use_direct: bool = True

    def __str__(self) -> str:
        return f'SolveRoutine({self.sorter},{self.precon},{self.iterative_solver}, {self.direct_solver})'
        
    def get_solver(self, A: lil_matrix, b: np.ndarray) -> Solver:
        """Returns the relevant Solver object given a certain matrix and source vector

        This is the default implementation for the SolveRoutine Class.
        
        Args:
            A (np.ndarray): The Matrix to solve for
            b (np.ndarray): the vector to solve for

        Returns:
            Solver: Returns the direct solver

        """
        return self.direct_solver
    
    def get_eig_solver(self, A: lil_matrix, b: lil_matrix, direct: bool = None) -> Solver:
        return self.direct_eig_solver
    
    def solve(self, A: np.ndarray | lil_matrix | csc_matrix, 
              b: np.ndarray, 
              solve_ids: np.ndarray,
              reuse: bool = False) -> np.ndarray:
        """ Solve the system of equations defined by Ax=b for x.

        Solve is the main function call to solve a linear system of equations defined by Ax=b.
        The solve routine will go through the required steps defined in the routine to tackle the problme.

        Args:
            A (np.ndarray | lil_matrix | csc_matrix): The (Sparse) matrix
            b (np.ndarray): The source vector
            solve_ids (np.ndarray): A vector of ids for which to solve the problem. For EM problems this
            implies all non-PEC degrees of freedom.
            reuse (bool): Whether to reuse the existing factorization if it exists.

        Returns:
            np.ndarray: The resultant solution.
        """
        solver = self.get_solver(A, b)

        NF = A.shape[0]
        NS = solve_ids.shape[0]

        solution = np.zeros((A.shape[0],), dtype=np.complex128)
        A = A.tocsc()

        logger.debug(f'Removing {NF-NS} prescribed DOFs ({NS} left)')

        Asel = A[np.ix_(solve_ids, solve_ids)]
        bsel = b[solve_ids]

        if solver.real_only:
            logger.debug('Converting to real matrix')
            Asel, bsel = complex_to_real_block(Asel, bsel)

        # SORT
        if solver.req_sorter and self.use_sorter:
            Asorted, bsorted = self.sorter.sort(Asel,bsel)
        else:
            Asorted, bsorted = Asel, bsel
        
        # Preconditioner
        if self.use_preconditioner and not self.iterative_solver.own_preconditioner:
            self.precon.init(Asorted, bsorted)

        start = time.time()
        x_solved, code = solver.solve(Asorted, bsorted, self.precon, reuse_factorization=reuse)
        end = time.time()
        logger.info(f'Time taken: {(end-start):.3f} seconds')
        logger.debug(f'O(N²) performance = {(NS**2)/((end-start)*1e6):.3f} MDoF/s')
        if self.use_sorter and solver.req_sorter:
            x = self.sorter.unsort(x_solved)
        else:
            x = x_solved
        
        if solver.real_only:
            logger.debug('Converting back to complex matrix')
            x = real_to_complex_block(x)
        
        solution[solve_ids] = x
        logger.debug('Solver complete!')
        if code:
            logger.debug('Solver code: {code}')
        return solution
    
    def eig(self, 
            A: np.ndarray | lil_matrix | csc_matrix, 
            B: np.ndarray, 
            solve_ids: np.ndarray,
            nmodes: int = 6,
            direct: bool = None,
            target_k0: float = None,
            which: str = 'LM') -> np.ndarray:
        """ Solve the system of equations defined by Ax=b for x.

        Solve is the main function call to solve a linear system of equations defined by Ax=b.
        The solve routine will go through the required steps defined in the routine to tackle the problme.

        Args:
            A (np.ndarray | lil_matrix | csc_matrix): The (Sparse) matrix
            b (np.ndarray): The source vector
            solve_ids (np.ndarray): A vector of ids for which to solve the problem. For EM problems this
            implies all non-PEC degrees of freedom.

        Returns:
            np.ndarray: The resultant solution.
        """
        solver = self.get_eig_solver(A, B, direct=direct)

        NF = A.shape[0]
        NS = solve_ids.shape[0]

        logger.debug(f'Removing {NF-NS} prescribed DOFs ({NS} left)')

        Asel = A[np.ix_(solve_ids, solve_ids)]
        Bsel = B[np.ix_(solve_ids, solve_ids)]
        
        eigen_values, eigen_modes = solver.eig(Asel, Bsel, nmodes, target_k0, which)
        
        return eigen_values, eigen_modes


class AutomaticRoutine(SolveRoutine):

    def get_solver(self, A: np.ndarray, b: np.ndarray) -> Solver:
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
        N = A.shape[0]
        if N > 5_000_000 or not self.use_direct:
            logger.warning('Using Iterative Solver due to large matrix size.' \
            'This simulation likely wont converge due to a lack of good preconditioner support.')
            self.use_preconditioner = True
            return self.iterative_solver
        else:
            self.use_preconditioner = False
            return self.direct_solver
        
    
### DEFAULTS

if 'arm' in platform.processor():
    direct_solver = SolverSuperLU()
else:
    direct_solver = SolverPardiso()

DEFAULT_ROUTINE = AutomaticRoutine(sorter=ReverseCuthillMckee(), 
                                   precon=ILUPrecon(), 
                                   iterative_solver=SolverGMRES(), 
                                   direct_solver=direct_solver,
                                   iterative_eig_solver=SolverARPACK(),
                                   direct_eig_solver=SolverLAPACK(),)