# [Written by Claude] All code in this module was written by Claude (AI).
"""SolverSparta.config(): SPARTA settings by name, analysis-only ones kept out of factorize()."""
import numpy as np
import pytest
import scipy.sparse as sp

from emerge._emerge.solvers.sparta import AVAILABLE, SolverSparta

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="SPARTA is not installed")


def _system(n: int = 300):
    A = sp.random(n, n, 0.01, random_state=1) + 5 * sp.eye(n)
    A = ((A + A.T) * (1 + 0.1j)).tocsc()
    b = (np.random.default_rng(0).standard_normal(n) + 0j)[:, None]
    return A, b


def test_analysis_only_settings_solve():
    A, b = _system()
    s = SolverSparta("t").set_options("CPU").config(nemin=32, refinement_steps=2, ordering="scalpel")
    x, _ = s.solve(A, b, None)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-10
    s.config(wide_cycles=0)
    assert s.sym is None  # a new setting discards the analysis
    x, _ = s.solve(A, b, None)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-10
    assert s.duplicate().settings == s.settings


@pytest.mark.parametrize("name", ["force_cpu", "hybrid", "mtype", "threads", "not_a_setting"])
def test_rejected_settings(name):
    with pytest.raises(ValueError):
        SolverSparta("t").config(**{name: 1})
