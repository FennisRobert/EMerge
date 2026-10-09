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


if __name__ == "__main__":
    test_solver_choice()
    test_basic_function()
