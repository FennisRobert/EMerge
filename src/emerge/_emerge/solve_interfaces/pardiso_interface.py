
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
import os
import sys
import ctypes
import re
import site
from ctypes.util import find_library
from enum import Enum
import numpy as np
from scipy import sparse # type: ignore
from pathlib import Path
from typing import Iterable, Iterator
import pickle
from loguru import logger

############################################################
#                        ERROR CODES                       #
############################################################

PARDISO_ERROR_CODES = {
     0: "No error.",
    -1: "Input inconsistent.",
    -2: "Not enough memory.",
    -3: "Reordering problem.",
    -4: "Zero pivot, numerical fac. or iterative refinement problem.",
    -5: "Unclassified (internal) error.",
    -6: "Preordering failed (matrix types 11(real and nonsymmetric), 13(complex and nonsymmetric) only).",
    -7: "Diagonal Matrix problem.",
    -8: "32-bit integer overflow problem.",
   -10: "No license file pardiso.lic found.",
   -11: "License is expired.",
   -12: "Wrong username or hostname.",
  -100: "Reached maximum number of Krylov-subspace iteration in iterative solver.",
  -101: "No sufficient convergence in Krylov-subspace iteration within 25 iterations.",
  -102: "Error in Krylov-subspace iteration.",
  -103: "Bread-Down in Krylov-subspace iteration",
}


############################################################
#               FINDING THE PARDISO DLL FILES              #
############################################################


#: Environment variable that overrides automatic searching

ENV_VAR = "EMERGE_PARDISO_PATH"


def _candidate_dirs() -> Iterable[Path]:
    """Return directories in which to look for MKL."""
    # Ordered from most to least likely
    seen: set[Path] = set()

    for p in (                         # likely “local” env first
        Path(sys.prefix),
        Path(getattr(sys, "base_prefix", sys.prefix)),
        Path(str(site.USER_BASE)),
        *(Path(x) for x in os.getenv("LD_LIBRARY_PATH", "").split(":") if x),
    ):
        if p not in seen:
            seen.add(p)
            yield p

def _search_mkl() -> Iterator[Path]:
    """Yield candidate MKL library paths, shortest first."""
    pattern = {
        "win32": r"^mkl_rt.*\.dll$",
        "darwin": r"^libmkl_rt(\.\d+)*\.dylib$",
        "linux": r"^libmkl_rt(\.so(\.\d+)*)?$",
    }.get(sys.platform, r"^libmkl_rt")

    regex = re.compile(pattern, re.IGNORECASE)

    for base in _candidate_dirs():
        for path in sorted(base.rglob("**/*mkl_rt*"), key=lambda p: len(str(p))):
            if regex.match(path.name):
                yield path

def cache_path_result(tag: str, compute_fn, force: bool = False) -> str:
    """
    Retrieve a cached Path object or compute it and store it.
    
    Parameters
    ----------
    tag : str
        Cache key.
    compute_fn : callable
        Callable that returns a Path.
    force : bool
        If True, bypass and overwrite the cache.
    """
    cache_dir = Path(__file__).parent / "__pycache__"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{tag}.pkl"

    if not force and cache_file.exists():
        with open(cache_file, "rb") as f:
            filename = pickle.load(f)
            logger.debug(f"Using cached MKL file: {filename}")
            return str(filename)

    result = compute_fn()
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return str(result)

def search_mkl() -> str:
    """Searches for the file path of the PARDISO MKL executable

    Returns:
        str: The filepath
    """
    logger.debug('Searching for MKL executable...')
    for candidate in _search_mkl():
        try:
            ctypes.CDLL(str(candidate))
        except OSError:
            continue  # try the next one
    logger.debug(f'Executable found: {candidate}')
    return str(candidate)

def load_mkl() -> ctypes.CDLL:
    """Locate and load **mkl_rt**; raise ImportError on failure."""
    # 1 explicit override
    override = os.getenv(ENV_VAR)
    if override:
        try:
            return ctypes.CDLL(override)
        except OSError as e:
            raise ImportError(f"{override!r} could not be loaded: {e}") from None

    # 2 system utility (cheap)
    lib = find_library("mkl_rt")
    if lib:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            pass
    
    # 2 system utility (cheap)
    lib = find_library("mkl_rt.1")
    if lib:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            pass

    # 3 filesystem walk (expensive, but last resort)
    try:
        filename = cache_path_result('mkl_file',search_mkl)
        return ctypes.CDLL(filename)
    except OSError:
        logger.warning('File name {filename} is no longer valid. Re-executing MKL search.')
        filename = cache_path_result('mkl_file',search_mkl,force=True)
        return ctypes.CDLL(filename)

    raise ImportError(
        "Shared library *mkl_rt* not found. "
        f"Set the environment variable {ENV_VAR} to its full path if it is in a non-standard location."
    )

    

############################################################
#                  ALL C-TYPE DEFINITIONS                 #
############################################################


class MKL_Complex16(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]

CINT64 = ctypes.c_int64
CINT32 = ctypes.c_int32
CFLOAT32 = ctypes.c_float
CFLOAT64 = ctypes.c_double

CPX16_P = ctypes.POINTER(MKL_Complex16)
CINT64_P = ctypes.POINTER(CINT64)
CINT32_P = ctypes.POINTER(CINT32)
CNONE_P = ctypes.POINTER(None)
CFLOAT32_P = ctypes.POINTER(CFLOAT32)
CFLOAT64_P = ctypes.POINTER(CFLOAT64)
VOID_SIZE = ctypes.sizeof(ctypes.c_void_p)

PT_A: type = CINT32
PT_B: type = np.int32

if VOID_SIZE == 8:
    PT_A = CINT64
    PT_B = np.int64
elif VOID_SIZE == 4:
    PT_A = CINT32
    PT_B = np.int32

def c_int(value: int):
    return ctypes.byref(ctypes.c_int32(value))

PARDISO_ARG_TYPES: tuple = (ctypes.POINTER(PT_A),CINT32_P,CINT32_P,
    CINT32_P,CINT32_P,CINT32_P,CNONE_P,CINT32_P,CINT32_P,
    CINT32_P,CINT32_P,CINT32_P,CINT32_P,CNONE_P,CNONE_P,CINT32_P,
)


############################################################
#                  PARDISO CONFIGURATIONS                 #
############################################################


class PARDISOMType(Enum):
    REAL_SYM_STRUCT = 1
    REAL_SYM_POSDEF = 2
    REAL_SYM_INDEF = -2
    COMP_SYM_STRUCT = 3
    COMP_HERM_POSDEF = 4
    COMP_HERM_INDEF = -4
    COMP_SYM = 6
    REAL_NONSYM = 11
    COMP_NONSYM = 13

class PARDISOPhase(Enum):
    CLEAR = -1
    NONE = 0
    SYMBOLIC_FACTOR = 11
    NUMERIC_FACTOR = 12
    NUMERIC_SOLVE = 33
    


############################################################
#                   GENERIC MATRIX CLASS                  #
############################################################


class SolveMatrix:
    def __init__(self, A: sparse.csr_matrix):
        A = A.tocsr()
        if not A.has_sorted_indices:
            A.sort_indices()

        if (A.getnnz(axis=1) == 0).any() or (A.getnnz(axis=0) == 0).any():
            raise ValueError('Matrix A is singular, because it contains empty rows or columns')

        if A.dtype in (np.float16, np.float32, np.float64):
            A = A.astype(np.float64)
        elif np.iscomplexobj(A):
            A = A.astype(np.complex128)
        self.mat: sparse.csr_matrix = A

        self.factorized_symb: bool = False
        self.factorized_num: bool = False

    
    def is_same(self, matrix: SolveMatrix):
        return id(self.mat) is id(matrix.mat)

    def same_structure(self, matrix: SolveMatrix):
        return np.all(matrix.mat.indices == self.mat.indices) & np.all(matrix.mat.indptr == self.mat.indptr)

    @property
    def zerovec(self):
        return np.zeros_like((self.mat.shape[0], 1), dtype=self.mat.dtype)


############################################################
#                   THE PARDISO INTERFACE                  #
############################################################

class PardisoInterface:

    def __init__(self):

        self.libmkl: ctypes.CDLL = load_mkl()
        self._pardiso_interface = self.libmkl.pardiso

        self._pardiso_interface.restype = None
        self._pardiso_interface.argtypes = PARDISO_ARG_TYPES
        
        self.PT = np.zeros(64, dtype=PT_B)
        self.IPARM = np.zeros(64, dtype=np.int32)
        self.PERM = np.zeros(0, dtype=np.int32)
        self.MATRIX_TYPE: PARDISOMType = None
        self.complex: bool = False
        self.mat_structure: np.ndarray = None
        self.message_level: int = 0

        self.matrix: SolveMatrix = None
        self.factored_matrix: SolveMatrix = None

        self.configure_solver()

    def _configure(self, A: sparse.csr_matrix) -> None:
        """Configures the solver for the appropriate data type (float/complex)

        Args:
            A (sparse.csr_matrix): The sparse matrix to solve
        """
        if np.iscomplexobj(A):
            self.MATRIX_TYPE = PARDISOMType.COMP_SYM_STRUCT
            self.complex = True
        else:
            self.MATRIX_TYPE = PARDISOMType.REAL_SYM_STRUCT
            self.complex = False

    def _prepare_B(self, b: np.ndarray | sparse.sparray) -> np.ndarray:
        """Fixes the forcing-vector for the solution process

        Args:
            b (np.ndarray): The forcing vector in Ax=b

        Returns:
            np.ndarray: The prepared forcing-vector
        """
        if sparse.issparse(b):
            b = b.todense() # type: ignore
        
        if np.iscomplexobj(b):
            b = b.astype(np.complex128)
        else:
            b = b.astype(np.float64)
        return b
    
    def symbolic(self, A: sparse.csr_matrix) -> int:
        """Calls the Symbollic solve routinge

        Returns:
            int: The error code
        """
        
        self._configure(A)
        zerovec = np.zeros((A.shape[0], 1), dtype=A.dtype)
        _, error = self._call_solver(A, zerovec, phase=PARDISOPhase.SYMBOLIC_FACTOR)
        return error

    def numeric(self, A: sparse.csr_matrix) -> int:
        """Calls the Numeric solve routine

        Returns:
            int: The error code
        """

        self._configure(A)
        zerovec = np.zeros((A.shape[0], 1), dtype=A.dtype)
        _, error = self._call_solver(A, zerovec, phase=PARDISOPhase.NUMERIC_FACTOR)
        return error
        
    def solve(self, A: sparse.csr_matrix, b: np.ndarray) -> tuple[np.ndarray, int]:
        """ Solves the linear problem Ax=b with PARDISO

        Args:
            A (sparse.csr_matrix): The A-matrix
            b (np.ndarray): The b-vector

        Returns:
            tuple[np.ndarray, int]: The solution vector x, and error code
        """
        self._configure(A)
        b = self._prepare_B(b)
        x, error = self._call_solver(A, b, phase=PARDISOPhase.NUMERIC_SOLVE)
        return x, error

    def configure_solver(self, 
        perm_algo: int = 3,
        nthreads: int | None = None,
        user_perm: int = 0,
        n_refine_steps: int = 0,
        pivot_pert: int = 13,
        weighted_matching: int = 2):
        """Configures the solver

        Args:
            perm_algo (int, optional): The permutation algorithm. Defaults to 3.
            nthreads (int, optional): The number of threads (Must be greater than OMP_NUM_THREADS). Defaults to None.
            user_perm (int, optional): 1, if a user permuation is provided (not supported yet). Defaults to 0.
            n_refine_steps (int, optional): Number of refinement steps. Defaults to 0.
            pivot_pert (int, optional): _description_. Defaults to 13.
            weighted_matching (int, optional): weighted matching mode. Defaults to 2.
        """
        if nthreads is None:
            nthreads = int(os.environ.get('OMP_NUM_THREADS', default="4"))

        self.IPARM[1] = perm_algo
        self.IPARM[2] = nthreads
        self.IPARM[4] = user_perm
        self.IPARM[7] = n_refine_steps
        self.IPARM[9] = pivot_pert
        self.IPARM[12] = weighted_matching

    def _call_solver(self, A: sparse.csr_matrix, b: np.ndarray, phase: PARDISOPhase) -> tuple[np.ndarray, int]:
        """Calls the PARDISO solver on linear problem Ax=b

        Args:
            A (sparse.csr_matrix): The A-matrix
            b (np.ndarray): The b-vector
            phase (PARDISOPhase): The solution phase

        Returns:
            tuple[np.ndarray, int]: The solution vector x and error code.
        """

        # Declare the empty vector
        x = np.zeros_like(b)
        error = ctypes.c_int32(0)
        
        # Up the pointers as PARDISO uses [1,...] indexing
        A_index_pointers = A.indptr + 1
        A_indices = A.indices + 1

        # Define the appropriate data type (complex vs real)

        if self.complex:
            VALUE_P = A.data.ctypes.data_as(CPX16_P)
            RHS_P = b.ctypes.data_as(CPX16_P)
            X_P = x.ctypes.data_as(CPX16_P)
        else:
            VALUE_P = A.data.ctypes.data_as(CFLOAT64_P)
            RHS_P = b.ctypes.data_as(CFLOAT64_P) # type: ignore
            X_P = x.ctypes.data_as(CFLOAT64_P) # type: ignore

        # Calls the pardiso function
        self._pardiso_interface(
            self.PT.ctypes.data_as(ctypes.POINTER(PT_A)),
            c_int(1),
            c_int(1),
            c_int(self.MATRIX_TYPE.value),
            c_int(phase.value),
            c_int(A.shape[0]),
            VALUE_P,
            A_index_pointers.ctypes.data_as(CINT32_P),
            A_indices.ctypes.data_as(CINT32_P),
            self.PERM.ctypes.data_as(CINT32_P),
            c_int(1),
            self.IPARM.ctypes.data_as(CINT32_P),
            c_int(self.message_level),
            RHS_P,
            X_P,
            ctypes.byref(error))
        
        # Returns the solution vector plus error code
        return np.ascontiguousarray(x), error.value

    def get_error(self, error: int) -> str:
        """Returns the PARDISO error description string given an error number

        Args:
            error (int): The error number

        Returns:
            str: A description string
        """
        return PARDISO_ERROR_CODES[error]

    def clear_memory(self):
        """Clear the memory of this solver plus the PARDISO process.
        """
        self.factorized_A = None
        self.matrix = None
        A = sparse.csr_matrix((0,0), dtype=np.complex128)
        b = np.zeros(0, dtype=np.complex128)
        _, error = self._call_solver(A, b, phase=PARDISOPhase.CLEAR)
        