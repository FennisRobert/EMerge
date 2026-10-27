from emerge._emerge.solver import (
    AutomaticRoutine,
    EMSolver,
    SolveRoutine,
)
from emerge import cleanup
import numpy as np
from scipy.sparse import csc_matrix, eye

# --- Create a small trial matrix ----
M = csc_matrix(np.eye(5).astype(np.complex128))
b = np.arange(5).astype(np.complex128).reshape(5, 1)
ids = np.arange(5)

# --- Create a big trial matrix ------
Mbig = eye(20000).tocsc().astype(np.complex128)
bbig = np.arange(20000).astype(np.complex128)
ids_big = np.arange(20000)


def test_basic_function():
    # Test a basic solve
    DEFAULT_ROUTINE = AutomaticRoutine()
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    DEFAULT_ROUTINE.reset()
    assert np.array_equiv(x, b)
    cleanup()


def test_solver_choice():
    DEFAULT_ROUTINE = AutomaticRoutine()
    # Test if hard coding SUPERLU works
    DEFAULT_ROUTINE.set_solver(EMSolver.SUPERLU)
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    assert report.solver == "SolverSuperLU"
    DEFAULT_ROUTINE.reset()

    # Test if hard coding UMFPACK works
    DEFAULT_ROUTINE.set_solver(EMSolver.UMFPACK)
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    assert report.solver == "SolverUMFPACK"
    DEFAULT_ROUTINE.reset()

    # Should still use UMFPACK
    DEFAULT_ROUTINE._configure_routine("MP")
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    assert report.solver == "SolverUMFPACK"
    DEFAULT_ROUTINE.reset()

    # Multi Threaded should use SUPERLU
    DEFAULT_ROUTINE._configure_routine("MT")
    x, report = DEFAULT_ROUTINE.solve(Mbig, bbig, ids_big)
    assert report.solver == "SolverSuperLU"
    DEFAULT_ROUTINE.reset()

    # Multi Processing should use UMFPACK on small problems
    DEFAULT_ROUTINE._configure_routine("MP")
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    assert report.solver == "SolverUMFPACK"
    DEFAULT_ROUTINE.reset()

    # Multi Processing should use UMFPACK on big problems
    DEFAULT_ROUTINE._configure_routine("MP")
    x, report = DEFAULT_ROUTINE.solve(Mbig, bbig, ids_big)
    assert report.solver == "SolverUMFPACK"
    DEFAULT_ROUTINE.reset()

    # Should Pick SuperLU if UMFPACK is Disabled.
    DEFAULT_ROUTINE.set_solver(EMSolver.UMFPACK)
    DEFAULT_ROUTINE.disable(EMSolver.UMFPACK)
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    assert report.solver == "SolverSuperLU"
    DEFAULT_ROUTINE.reset()

    # Should Pick SuperLU if UMFPACK is Disabled.
    DEFAULT_ROUTINE.disable(EMSolver.UMFPACK)
    x, report = DEFAULT_ROUTINE.solve(M, b, ids)
    assert report.solver == "SolverSuperLU"
    DEFAULT_ROUTINE.reset()
    cleanup()

def test_aux_functionality():
    DEFAULT_ROUTINE = AutomaticRoutine()
    assert isinstance(DEFAULT_ROUTINE.duplicate(), SolveRoutine)
    assert isinstance(DEFAULT_ROUTINE.duplicate(), AutomaticRoutine)


def test_duplicate_gives_independent_routines():
    """Worker threads of a sweep must not share a routine or solver objects (shared PARDISO handles crashed)."""
    routine = AutomaticRoutine()
    forced = next(iter(routine.solvers.values()))
    routine.set_solver(forced)
    a, b = routine.duplicate(), routine.duplicate()
    assert a is not routine and b is not a
    for key, solver in routine.solvers.items():
        assert a.solvers[key] is not solver and a.solvers[key] is not b.solvers[key]
    assert a.forced_solver[0] is not forced and a.forced_solver[0] is not b.forced_solver[0]


def test_physics_share_one_routine():
    """set_solver() on the Simulation applies to both physics, and every Simulation gets a fresh routine."""
    import emerge as em

    sim1 = em.Simulation("RoutineA", loglevel="ERROR")
    assert sim1.hc.solveroutine is sim1.mw.solveroutine
    routine1 = sim1.mw.solveroutine
    em.cleanup()
    sim2 = em.Simulation("RoutineB", loglevel="ERROR")
    assert sim2.mw.solveroutine is not routine1
    em.cleanup()


# [Written by Claude]
def test_nthreads_reaches_solver(monkeypatch):
    """SolveRoutine.solve(nthreads=...) installs the thread count on the solver that does the solve."""
    from emerge._emerge.solver import SolverPardiso, SolverSuperLU

    seen = []
    monkeypatch.setattr(SolverSuperLU, "set_nthreads", lambda self, n: seen.append(n))
    routine = AutomaticRoutine()
    routine.set_solver(EMSolver.SUPERLU)
    x, report = routine.solve(M, b, ids, nthreads=3)
    x, report = routine.solve(M, b, ids)
    routine.reset()
    assert np.array_equiv(x, b)
    assert seen == [3]  # None keeps the solver's own setting

    # PARDISO keeps the count until its interface exists, which then starts with it
    pardiso = SolverPardiso("")
    pardiso.set_nthreads(4)
    assert pardiso.nthreads == 4 and pardiso.solver is None


# [Written by Claude]
def test_sparta_nthreads(monkeypatch):
    """SPARTA takes the thread count in analyze() and analyzes again when it changes. A count set on the
    solver itself survives a solve without nthreads."""
    import pytest
    from emerge._emerge import solver as solver_module

    if not solver_module._SPARTA_AVAILABLE:
        pytest.skip("SPARTA not installed")
    sparta = solver_module.sparta
    if solver_module.Version(sparta.__version__) < solver_module.Version("0.3.0"):
        pytest.skip("SPARTA < 0.3.0 has no thread count")

    calls = []
    analyze = sparta.analyze
    monkeypatch.setattr(sparta, "analyze", lambda *a, **kw: calls.append(kw.get("threads")) or analyze(*a, **kw))

    n = 400
    A = (eye(n) * 4 + eye(n, k=1) + eye(n, k=-1)).tocsc().astype(np.complex128)
    rhs = np.ones((n, 1), dtype=np.complex128)
    routine = AutomaticRoutine()
    routine.set_solver(EMSolver.SPARTA)
    routine.forced_solver[0].set_options("CPU")
    for nthreads in (2, 2, 1, None):
        x, report = routine.solve(A, rhs, np.arange(n), nthreads=nthreads)
        assert report.solver == "SolverSparta"
        assert np.allclose(A @ x, rhs)
    routine.reset()
    assert calls == [2, 1]

    solver = routine.forced_solver[0].set_nthreads(3)
    routine.solve(A, rhs, np.arange(n))
    routine.reset()
    assert solver.nthreads == 3 and calls[-1] == 3


if __name__ == "__main__":
    test_solver_choice()
    test_basic_function()
