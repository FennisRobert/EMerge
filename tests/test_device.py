# [Written by Claude] The solver device of a solve routine: set_device("CPU" / "GPU" / "HYBRID").
import pytest
import scipy.sparse as sp
from loguru import logger

import emerge as em
from emerge._emerge.solveroutine import AutomaticRoutine
from emerge._emerge.solvers import MatrixType, SolverSparta, SolverCuDSS

needs_sparta = pytest.mark.skipif(not SolverSparta.available, reason="SPARTA is not installed")

# Large enough for the automatic routine to use a fast direct solver; pick_solver only looks at the shape
A = sp.identity(10_001, dtype=complex, format="csc")
b = A[:, :1].toarray()


def pick(routine: AutomaticRoutine):
    return routine.pick_solver(A, b, MatrixType.COMPLEX_SYMMETRIC)


@pytest.fixture
def warnings():
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(handler)


def test_default_is_cpu():
    routine = AutomaticRoutine()
    assert routine.device == "CPU"
    assert not isinstance(pick(routine), SolverCuDSS)


@needs_sparta
def test_cpu_prefers_sparta_cpu():
    routine = AutomaticRoutine()
    solver = pick(routine)
    assert isinstance(solver, SolverSparta) and solver.mode == "CPU"


def test_invalid_device():
    with pytest.raises(ValueError):
        AutomaticRoutine().set_device("TPU")  # type: ignore


def test_case_insensitive():
    routine = AutomaticRoutine()
    routine.set_device("cpu")  # type: ignore
    assert routine.device == "CPU"


def test_gpu_without_gpu_solver_warns_and_uses_cpu(monkeypatch, warnings):
    monkeypatch.setattr(SolverSparta, "gpu", False)
    monkeypatch.setattr(SolverCuDSS, "available", False)
    routine = AutomaticRoutine()
    routine.set_device("GPU")
    assert any("No GPU solver" in m for m in warnings)
    solver = pick(routine)
    assert not isinstance(solver, SolverCuDSS)
    if isinstance(solver, SolverSparta):
        assert solver.mode == "CPU"


def test_hybrid_without_sparta_gpu_warns(monkeypatch, warnings):
    monkeypatch.setattr(SolverSparta, "gpu", False)
    monkeypatch.setattr(SolverCuDSS, "available", False)
    AutomaticRoutine().set_device("HYBRID")
    assert any("HYBRID" in m for m in warnings)


@needs_sparta
@pytest.mark.parametrize("device", ["GPU", "HYBRID"])
def test_gpu_devices_use_sparta(monkeypatch, warnings, device):
    monkeypatch.setattr(SolverSparta, "gpu", True)
    routine = AutomaticRoutine()
    routine.set_device(device)
    assert not warnings
    solver = pick(routine)
    assert isinstance(solver, SolverSparta) and solver.mode == device
    # Back to the CPU path
    routine.set_device("CPU")
    assert solver.mode == "CPU"


@needs_sparta
def test_gpu_device_not_used_by_thread_workers(monkeypatch):
    monkeypatch.setattr(SolverSparta, "gpu", True)
    routine = AutomaticRoutine()
    routine.set_device("GPU")
    routine._configure_routine("MT")
    assert routine._device_solver() is None


@needs_sparta
def test_duplicate_and_reset(monkeypatch):
    monkeypatch.setattr(SolverSparta, "gpu", True)
    routine = AutomaticRoutine()
    routine.set_device("HYBRID")
    copy = routine.duplicate()
    assert copy.device == "HYBRID" and copy.solvers[SolverSparta].mode == "HYBRID"
    routine.reset()
    assert routine.device == "HYBRID"
    routine.reset(reset_solver_preference=True)
    assert routine.device == "CPU" and routine.solvers[SolverSparta].mode == "CPU"


def test_simulation_set_device():
    model = em.Simulation("DeviceTest")
    model.set_device("GPU")
    assert model.mw.solveroutine.device == "GPU"
    assert model.hc.solveroutine.device == "GPU"
    em.cleanup()
