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
import warnings
from loguru import logger
from packaging import version
# Catch the Cuda warning and print it with Loguru.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import cupy as cp
    cpsparse = None
    if version.parse(cp.__version__) >= version.parse("14.0"):
        import cupyx.scipy.sparse as cpsparse
    elif version.parse(cp.__version__) >= version.parse("13.0"):
        import cupy.sparse as cpsparse
    else:
        logger.warning(f"Unsupported CuPy version {cp.__version__}, sparse not available")
    for warn in w:
        logger.debug(f"{warn.category.__name__}: {warn.message}")

import nvmath.bindings.cudss as cudss # ty: ignore
from nvmath import CudaDataType # ty: ignore

from scipy.sparse import csr_matrix
import numpy as np
from typing import Literal

############################################################
#                         CONSTANTS                        #
############################################################

# nvmath-python 1.0 renamed AlgType (ALG_DEFAULT/ALG_1/2/3) to ReorderingAlg with descriptive names.
if hasattr(cudss, "ReorderingAlg"):
    ReorderingAlg = cudss.ReorderingAlg
    ALG_NEST_DISS_METIS = ReorderingAlg.DEFAULT
    ALG_COLAMD = ReorderingAlg.COLAMD
    ALG_COLAMD_BLOCK_TRI = ReorderingAlg.BTF_COLAMD
    ALG_AMD = ReorderingAlg.AMD
else:
    ReorderingAlg = cudss.AlgType
    ALG_NEST_DISS_METIS = ReorderingAlg.ALG_DEFAULT
    ALG_COLAMD = ReorderingAlg.ALG_1
    ALG_COLAMD_BLOCK_TRI = ReorderingAlg.ALG_2
    ALG_AMD = ReorderingAlg.ALG_3

FLOAT64 = CudaDataType.CUDA_R_64F
FLOAT32 = CudaDataType.CUDA_R_32F
COMPLEX128 = CudaDataType.CUDA_C_64F
COMPLEX64 = CudaDataType.CUDA_C_32F
INT64 = CudaDataType.CUDA_R_64I
INT32 = CudaDataType.CUDA_R_32I

INDEX_BASE = cudss.IndexBase.ZERO

def _c_pointer(arry) -> int:
    return int(arry.data.ptr)


def _matrix_create_csr(nrows, ncols, nnz, row_start, row_end, col_indices, values,
                       index_type, value_type, mtype, mview, index_base) -> int:
    """cudss.matrix_create_csr for both nvmath APIs: nvmath >= 1.0 has an extra offset_type argument
    (the integer type of the row pointers) before index_type."""
    try:
        return cudss.matrix_create_csr(nrows, ncols, nnz, row_start, row_end, col_indices, values,
                                       index_type, index_type, value_type, mtype, mview, index_base)
    except TypeError:
        return cudss.matrix_create_csr(nrows, ncols, nnz, row_start, row_end, col_indices, values,
                                       index_type, value_type, mtype, mview, index_base)


############################################################
#                    GPU PRECISION CHECK                   #
############################################################

# Fallback FP32:FP64 throughput ratios by compute capability, for CuPy versions that do not report
# singleToDoublePrecisionPerfRatio. Data center GPUs (P100, V100, A100, H100, B200) run FP64 at half
# rate; consumer and workstation GPUs at 1/32 or 1/64.
_FP64_RATIO_BY_CC = {
    (6, 0): 2, (6, 1): 32, (6, 2): 32,
    (7, 0): 2, (7, 2): 32, (7, 5): 32,
    (8, 0): 2, (8, 6): 64, (8, 7): 64, (8, 9): 64,
    (9, 0): 2,
    (10, 0): 2, (12, 0): 64,
}


def gpu_fp64_ratio(device: int | None = None) -> tuple[int | None, str]:
    """Returns (FP32:FP64 throughput ratio, device name) of a CUDA device. The ratio is None if unknown."""
    dev = cp.cuda.Device().id if device is None else device
    props = cp.cuda.runtime.getDeviceProperties(dev)
    name = props.get("name", b"")
    name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
    ratio = props.get("singleToDoublePrecisionPerfRatio")
    if not ratio:
        ratio = _FP64_RATIO_BY_CC.get((props.get("major"), props.get("minor")))
    return (int(ratio) if ratio else None), name



############################################################
#                         INTERFACE                        #
############################################################

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class CuDSSInterface(metaclass=Singleton):
    """the CuDSSInterface class implements the nvmath bindings and cupy
    control for EMerge.
    """
    AlgType = ReorderingAlg

    def __init__(self):
        self.A_cu = None
        self.b_cu = None
        self.x_cu = None
        self.A_cobj = None
        self.b_cobj = None
        self.x_cobj = None
        self.A_pattern = None

        self._handle = cudss.create()
        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)

        self.MTYPE = cudss.MatrixType.GENERAL
        self.MVIEW = cudss.MatrixViewType.FULL
        self.RALG = ALG_NEST_DISS_METIS
        self.VTYPE = CudaDataType.CUDA_R_64F

        self._INDPTR = None
        self._ROW_START: int | None = None
        self._ROW_END: int | None = None
        self._IND = None
        self._VAL = None
        self._NNZ: int | None = None
        self._COMP: bool = True
        self._PRES: int = 2
        self._COL_IDS = None
        self._csym: bool = True
        self._initialized = False
        self._factorized: bool = False  # the first numeric phase after ANALYSIS must be FACTORIZATION

        self._apply_reordering()

    def _apply_reordering(self) -> None:
        param = cudss.ConfigParam.REORDERING_ALG
        dtype = cudss.get_config_param_dtype(int(param))
        reorder_alg = np.array(self.RALG, dtype=dtype)

        cudss.config_set(
            self._config,
            int(param),
            reorder_alg.ctypes.data,
            reorder_alg.nbytes
        )

    def clear_memory(self):
        # TODO: Implement this
        pass
    
    def set_algorithm(self, alg_type: Literal['METIS','COLAMD','COLAMD_BT','AMD']):
        """Define fill-in reduction column permuation algorithm. The options are:

         - "METIS" (Default) = NVidia's own Nested Dissection METIS sorter
         - "COLAMD" = Column approximate minimum degree
         - "COLAMD_BT" = Column Approximate Minimum Degree Block Triangular ("COLAM_BT" is also accepted)
         - "AMD" = Approximate Minimum Degree

        Args:
            alg_type (str): The chosen type
        """
        if alg_type=='METIS':
            self.RALG = ALG_NEST_DISS_METIS
        elif alg_type =='COLAMD':
            self.RALG = ALG_COLAMD
        elif alg_type in ('COLAMD_BT', 'COLAM_BT'):
            self.RALG = ALG_COLAMD_BLOCK_TRI
        elif alg_type == 'AMD':
            self.RALG = ALG_AMD
        else:
            logger.warning(f'Algorithm type {alg_type} is not of the chosen set. Ignoring setting.')
            return
        self._apply_reordering()
    
    def init_type(self):
        """Initializes the value data type of the solver (float vs complex, single vs double).
        """
        if self._PRES == 1:
            if self._COMP:
                self.c_dtype = cp.complex64
                self.VTYPE = COMPLEX64
            else:
                self.c_dtype = cp.float32
                self.VTYPE = FLOAT32
        else:
            if self._COMP:
                self.c_dtype = cp.complex128
                self.VTYPE = COMPLEX128
            else:
                self.c_dtype = cp.float64
                self.VTYPE = FLOAT64

    def submit_matrix(self, A: csr_matrix):
        """Sets the given csr_matrix as the matrix to be solved.

        Args:
            A (csr_matrix): The csr_format matrix for the problem Ax=b
        """
        self.N = A.shape[0]

        if self._csym:
            self.MTYPE = cudss.MatrixType.GENERAL
        else:
            self.MTYPE = cudss.MatrixType.GENERAL
        
        if np.iscomplexobj(A):
            self._COMP = True
        else:
            self._COMP = False
        
        self.init_type()

        self.A_cu = cpsparse.csr_matrix(A).astype(self.c_dtype)

        self._INDPTR = cp.ascontiguousarray(self.A_cu.indptr.astype(cp.int32))
        self._IND = cp.ascontiguousarray(self.A_cu.indices.astype(cp.int32))
        self._VAL = cp.ascontiguousarray(self.A_cu.data)
        self._NNZ = int(self._VAL.size)
        self._ROW_START = self._INDPTR[:-1]
        self._ROW_END = self._INDPTR[1:]
        self._COL_IDS = self.A_cu.indices.astype(cp.int32)

    def submit_vector(self, b: np.ndarray):
        """Submits the dense vector b to be solved.

        Args:
            b (np.ndarray): The dense vector for the problem Ax=b
        """
        self.b_cu = cp.array(b).astype(self.c_dtype)
    
    def create_solvec(self):
        """Initializes a solution vector that the nvmath binding can access.
        """
        self.x_cu = cp.empty_like(self.b_cu)

    def _update_dss_data(self):
        """Updates the currently defined matrix data into the existing memory.ALG_AMD
        """
        cudss.matrix_set_values(self.A_cobj, _c_pointer(self._VAL))
        
    def _update_rhs_data(self):
        self.b_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.b_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))
        self.x_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.x_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))

    def _create_dss_data(self):
        """Creates a new memory slot for the CSR matrix of the matrix A"""
        # Plain CSR: row_end is NULL (cuDSS >= 0.8 rejects separate row_start/row_end arrays).
        self.A_cobj = _matrix_create_csr(
            self.N,self.N,self._NNZ,
            _c_pointer(self._INDPTR),
            0,
            _c_pointer(self._COL_IDS),
            _c_pointer(self._VAL),
            int(INT32), 
            int(self.VTYPE),
            int(self.MTYPE),
            int(self.MVIEW),
            int(INDEX_BASE),
        )
        self.b_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.b_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))

        self.x_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.x_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))

    def symbolic(self, A: csr_matrix) -> np.ndarray:
        """Executes symbolic LU factorization of Ax=b starting 

        Args:
            A (csr_matrix): The input sparse matrix
            b (np.ndarray): The solution vector b

        Returns:
            np.ndarray: The solved vector
        """
        self.submit_matrix(A)
        self.submit_vector(np.zeros((A.shape[0],), dtype=np.complex128))
        self.create_solvec()
        self._create_dss_data()
        self._symbolic()

    def numeric(self, A: csr_matrix | None) -> np.ndarray:
        """Executes numeric factorization of A.

        Args:
            A (csr_matrix): The input sparse matrix
            b (np.ndarray): The solution vector b

        Returns:
            np.ndarray: The solved vector
        """
        self.submit_matrix(A)
        self.submit_vector(np.zeros((A.shape[0],), dtype=np.complex128))
        self.create_solvec()
        self._update_dss_data()
        self._numeric(refactorize=self._factorized)
        self._factorized = True

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solves Ax=b only with a new b vector.

        Args:
            A (csr_matrix): The input sparse matrix
            b (np.ndarray): The solution vector b

        Returns:
            np.ndarray: The solved vector
        """
        self.submit_vector(b)
        self.create_solvec()
        self._update_rhs_data()
        return self._solve()

    def _symbolic(self):
        logger.trace('Executing symbolic factorization')
        self._factorized = False
        cudss.execute(self._handle, cudss.Phase.ANALYSIS, self._config, self._data, 
                      self.A_cobj, self.x_cobj, self.b_cobj)

    def _numeric(self, refactorize: bool = False):
        if refactorize:
            logger.trace('Refactoring matrix')
            phase = cudss.Phase.REFACTORIZATION
        else:
            phase = cudss.Phase.FACTORIZATION
        logger.trace('Executing numerical factorization')
        cudss.execute(self._handle, phase, self._config, self._data, 
                      self.A_cobj, self.x_cobj, self.b_cobj)
                      
    def _solve(self) -> np.ndarray:
        logger.trace('Solving matrix problem')
        cudss.execute(self._handle, cudss.Phase.SOLVE, self._config, self._data, 
                      self.A_cobj, self.x_cobj, self.b_cobj)
        cp.cuda.runtime.deviceSynchronize()
        x_host = cp.asnumpy(self.x_cu).ravel()
        return x_host
