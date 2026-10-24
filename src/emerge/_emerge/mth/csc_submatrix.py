# [Written by Claude] All code in this module was written by Claude (AI).
"""A[:, ids][ids, :] of a square CSC matrix in one parallel pass (emr.csc_submatrix)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix


def csc_submatrix(A, ids: np.ndarray):
    """The same matrix as A[:, ids][ids, :], built in one parallel pass when A is a square complex CSC matrix
    with 32 bit indices and ids are increasing; otherwise with scipy's fancy indexing."""
    ids = np.asarray(ids)
    if (
        getattr(A, "format", "") != "csc"
        or A.shape[0] != A.shape[1]
        or A.dtype != np.complex128
        or A.indices.dtype != np.int32
        or A.indptr.dtype != np.int32
        or ids.ndim != 1
        or ids.size == 0
        or ids.dtype.kind not in "iu"
        or np.any(np.diff(ids) <= 0)
        or ids[0] < 0
        or ids[-1] >= A.shape[0]
    ):
        return A[:, ids][ids, :]
    from emr.csc_submatrix import csc_submatrix_c16

    indptr, indices, data = csc_submatrix_c16(
        A.indptr, A.indices, np.ascontiguousarray(A.data), A.shape[0], ids.astype(np.int64)
    )
    return csc_matrix((data, indices, indptr), shape=(ids.size, ids.size))
