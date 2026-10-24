# [Written by Claude] All code in this module was written by Claude (AI).
"""csc_submatrix (removal of prescribed DoFs before the solve) against scipy's fancy indexing."""
import numpy as np
import scipy.sparse as sp

from emerge._emerge.mth.csc_submatrix import csc_submatrix


def _matrix(n: int, per: int, seed: int = 0) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)
    r = rng.integers(0, n, n * per)
    c = np.repeat(np.arange(n), per)
    v = rng.standard_normal(n * per) + 1j * rng.standard_normal(n * per)
    A = sp.csc_matrix((v, (r, c)), shape=(n, n))
    return (A + A.T + sp.eye(n)).tocsc()


def test_submatrix_matches_fancy_indexing():
    rng = np.random.default_rng(1)
    A = _matrix(5000, 12)
    for frac in (1.0, 0.9, 0.3):
        ids = np.sort(rng.choice(A.shape[0], int(frac * A.shape[0]), replace=False))
        ref = A[:, ids][ids, :]
        out = csc_submatrix(A, ids)
        assert out.format == "csc" and out.shape == ref.shape and out.has_sorted_indices
        assert np.array_equal(out.indptr, ref.indptr)
        assert np.array_equal(out.indices, ref.indices)
        assert np.array_equal(out.data, ref.data)


def test_submatrix_falls_back_to_scipy():
    A = _matrix(300, 5)
    ids = np.array([5, 3, 200, 7])  # not increasing
    assert abs(csc_submatrix(A, ids) - A[:, ids][ids, :]).max() == 0
    Ar = A.real.tocsc()  # real matrix
    ids = np.arange(0, 300, 2)
    assert abs(csc_submatrix(Ar, ids) - Ar[:, ids][ids, :]).max() == 0
    Ar = A.tocsr()  # not CSC
    assert abs(csc_submatrix(Ar, ids) - Ar[:, ids][ids, :]).max() == 0
