# [Written by Claude] em.set_threads() sets the thread count of the compiled kernels and the default of the solvers.
import numpy as np
import pytest
import emerge as em
from emr import mesh3d
from emerge._emerge.threads import solver_threads


@pytest.fixture(autouse=True)
def reset_threads():
    yield
    em.set_threads(None)


def test_set_and_reset():
    default = em.get_threads()
    em.set_threads(2)
    assert em.get_threads() == 2
    em.set_threads(None)
    assert em.get_threads() == default


@pytest.mark.parametrize("bad", [0, -1, 2.5, "4"])
def test_invalid_count(bad):
    with pytest.raises(ValueError):
        em.set_threads(bad)


def test_kernels_give_the_same_result_on_one_thread():
    rng = np.random.default_rng(1)
    tets = np.sort(rng.choice(5000, size=(4, 20000)), axis=0).astype(np.int64)
    edges, tris = mesh3d.extract_edges_tris(tets)
    em.set_threads(1)
    edges1, tris1 = mesh3d.extract_edges_tris(tets)
    assert np.array_equal(edges, edges1) and np.array_equal(tris, tris1)


def test_solver_thread_precedence():
    assert solver_threads(None) is None
    em.set_threads(3)
    assert solver_threads(None) == 3
    assert solver_threads(5) == 5
