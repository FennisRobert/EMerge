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

"""The Apple Accelerate sparse direct solver: the interface to emerge_aasds and the Solver."""
from __future__ import annotations
import time
from typing import Literal
import numpy as np
from scipy.sparse import csc_matrix  # type: ignore
from loguru import logger
from .solver_base import Solver, SolverType, SolveReport, _pfx

try:
    from emerge_aasds import Factorization, Symmetry, Scaling, Ordering  # type: ignore # noqa: F401
    AVAILABLE = True
except ModuleNotFoundError:
    logger.trace("AASDS Interface not found. (not harmful)")
    AVAILABLE = False
except ImportError:
    logger.debug("Tried to import an installed AASDS solver on a non Darwin system.")
    AVAILABLE = False


class AASDSInterface:
    """
    Apple Accelerate Sparse Solver Interface
    
    Uses Apple's native highly-optimized sparse direct solver
    
    Usage:
        ctx = AccelerateInterface()
        ctx.analyse(A)
        ctx.factorize(A)
        x, info = ctx.solve(b)
    
    Parameters
    ----------
    factorization : str, optional
        Factorization type: 'lu' (default, works for complex), 'qr', 'cholesky', 'ldlt'
    symmetry : str, optional
        Matrix symmetry: 'nonsymmetric' (default), 'symmetric', 'hermitian'
        Note: For complex symmetric matrices (A=A^T, not A=A^H), use 'nonsymmetric'
              Accelerate's 'symmetric' mode may assume A=A^H for complex
    verbose : int, optional
        Verbosity level (0=silent, 1=timing info)
    
    Recommended settings for EM problems:
        - Complex symmetric: factorization='lu', symmetry='nonsymmetric'
        - Real SPD: factorization='cholesky', symmetry='symmetric'
    """
    def __init__(self, verbose=0):
        self._solver = None
        self._factorization = Factorization.LU
        self._usfactorization = Factorization.LU
        self._csym: bool = True
        
        self._factored = False
        self.verbose = verbose
    
    @property
    def _symmetry(self) -> Symmetry:
        if self._csym:
            return Symmetry.SYMMETRIC
        else:
            return Symmetry.NONSYMMETRIC
    
    @property
    def factorization(self):
        if self._symmetry is Symmetry.SYMMETRIC:
            return self._factorization
        else:
            return self._usfactorization
        
    def initialize(self): 
        from emerge_aasds.interface import AccelerateInterface

        self._solver = AccelerateInterface(
            factorization=self.factorization,
            ordering=Ordering.MTMETIS,
            pivot_tolerance=0.001,
            symmetry=self._symmetry,
            verbose=0  # We handle our own verbosity
        )

    def analyse(self, A):
        """
        Symbolic factorization
        
        Parameters
        ----------
        A : scipy.sparse matrix
            Sparse matrix to analyze
        """
        self.initialize()
        t0 = time.time()
        
        self._solver.analyse(A)
        
        if self.verbose > 0:
            # ALLOWED PRINT
            logger.info(f"Analyse: {time.time()-t0:.3f}s")
        
        self._factored = False
    
    def factorize(self, A):
        """
        Numeric factorization
        
        Parameters
        ----------
        A : scipy.sparse matrix
            Matrix with same sparsity pattern as analyse()
        """
        if self._solver is None:
            raise RuntimeError("Call analyse() first")
        
        t0 = time.time()
        
        try:
            self._solver.factorize(A)
        except Exception as e:
            # Factorization failed - provide helpful error message
            error_msg = f"Factorization failed with {self._factorization}: {e}"
            
            if self._factorization == 'ldlt':
                error_msg += "\n  Hint: LDLT requires symmetric indefinite matrices."
                error_msg += "\n        For complex symmetric EM problems, try 'qr' instead."
            elif self._factorization == 'cholesky':
                error_msg += "\n  Hint: Cholesky requires symmetric POSITIVE DEFINITE matrices."
                error_msg += "\n        For indefinite matrices, try 'ldlt' or 'qr'."
            elif self._factorization == 'lu':
                error_msg += "\n  Hint: LU factorization failed - matrix may be singular."
                error_msg += "\n        Try 'qr' for more robust factorization."
            
            raise RuntimeError(error_msg)
        
        if self.verbose > 0:
            logger.info(f"factorize: {time.time()-t0:.3f}s")
        
        self._factored = True
    
    def solve(self, b):
        """
        Solve system using factorization
        
        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector or matrix
        
        Returns
        -------
        x : numpy.ndarray
            Solution vector or matrix
        info : dict or None
            Solver info (for compatibility with MUMPS interface)
        """
        if not self._factored:
            raise RuntimeError("Call factorize() first")
        
        t0 = time.time()
        x, info = self._solver.solve(b)
        
        if self.verbose > 0:
            logger.info(f"Run: {time.time()-t0:.3f}s")
        
        return x, info
    
    def destroy(self):
        """Cleanup solver resources"""
        if self._solver is not None:
            self._solver.destroy()
            self._solver = None
        self._factored = False
    
    def __del__(self):
        self.destroy()
    
    def __repr__(self):
        status = "factored" if self._factored else "not factored"
        return f"AccelerateInterface(factorization='{self._factorization}', symmetry='{self._csym}', {status})"


# Convenience function matching SPOOLES pattern
def create_accelerate_interface(factorization='lu', symmetry='nonsymmetric', verbose=0):
    """
    Create Accelerate solver interface
    
    Parameters
    ----------
    factorization : str, optional
        'lu' (default, works for complex general matrices)
        'qr' (robust but may crash on complex matrices in macOS 26)
        'cholesky' (symmetric positive definite only, real matrices)
        'ldlt' (symmetric indefinite, real matrices only)
    
    symmetry : str, optional
        'nonsymmetric' (default, safest for complex matrices)
        'symmetric' (A = A^T, use for real symmetric only)
        'hermitian' (A = A^H, for complex Hermitian)
    
    verbose : int, optional
        0 = silent, 1 = timing info
    
    Returns
    -------
    solver : AccelerateInterface
        Configured solver instance
    
    Examples
    --------
    For complex symmetric EM problems (RECOMMENDED):
    >>> solver = create_accelerate_interface('lu', symmetry='nonsymmetric', verbose=1)
    >>> solver.analyse(A)
    >>> solver.factorize(A)
    >>> x, info = solver.solve(b)
    
    For real symmetric positive definite (fastest):
    >>> solver = create_accelerate_interface('cholesky', symmetry='symmetric', verbose=1)
    
    Notes
    -----
    - LU is most reliable for complex matrices on macOS 26
    - QR/LDLT may crash on complex matrices (Apple Accelerate limitation)
    - For complex symmetric (A=A^T), use symmetry='nonsymmetric' to avoid issues
    """
    return AASDSInterface(factorization=factorization, symmetry=symmetry, verbose=verbose)


class SolverAASDS(Solver):
    """Implements the Apple Accelerate Sparse Direct solver.

    Requires macOS and the emerge-aasds package (installed with EMerge on macOS).
    """

    available = AVAILABLE
    requires = "macOS and the emerge-aasds package (installed with EMerge on macOS)"

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
        self.aasds._factorization = Factorization(self.sym_fact)
        self.aasds._usfactorization = Factorization(self.unsym_fact)

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
    
        logger.trace(f"{_pfx(self.pre, id)} Executing numeric factorization.")
        self.aasds.factorize(A)
        self.A = A

        logger.trace(f"{_pfx(self.pre, id)} Solving linear system.")
        x = np.zeros_like(b)
        for i in range(x.shape[1]):
            logger.trace(f"{_pfx(self.pre, id)} Solve RHS {i}.")
            x[:, i], _ = self.aasds.solve(b[:, i])

        return x, SolveReport(solver=str(self), exit_code=0)
