# [Written by Claude]
"""The threaded job dispatch must reset the solve routine of every worker thread on that same thread.
scipy's SuperLU frees a factorization only in the thread that created it, so a routine that is dropped
or reset in another thread keeps its factorization allocated after every batch of a sweep."""
import threading
from types import SimpleNamespace

import pytest

from emerge._emerge.physics.microwave.microwave_3d import Microwave3D


class _Report:
    def add(self, **kw):
        pass


class _Routine:
    def __init__(self, created):
        self.resets = []
        self.thread = None
        created.append(self)

    def _configure_routine(self, parallel, thread_nr=0):
        return self

    def solve(self, A, b, ids, **kw):
        self.thread = threading.get_ident()
        if A == "fail":
            raise RuntimeError("solver failed")
        return b, _Report()

    def reset(self):
        self.resets.append(threading.get_ident())


class _Parent:
    def __init__(self):
        self.created = []
        self.lock = threading.Lock()

    def duplicate(self):
        with self.lock:
            return _Routine(self.created)


class _Job:
    def __init__(self, i, A="ok"):
        self.id, self.mtype, self._precon, self.k0, self.A = i, None, None, None, A

    def get_Ab(self):
        return self.A, self.id, None, {}

    def submit_solution(self, solution, report):
        self.solution = solution


def _dispatch(jobs, parent):
    sim = SimpleNamespace(solveroutine=parent, _direct=True)
    return Microwave3D._dispatch_jobs(sim, jobs, parallel=True, multi_processing=False, n_workers=3)


def test_worker_routines_are_reset():
    parent = _Parent()
    jobs = _dispatch([_Job(i) for i in range(12)], parent)
    assert [j.solution for j in jobs] == list(range(12))
    assert parent.created
    assert all(r.resets == [r.thread] for r in parent.created)


def test_worker_routines_are_reset_when_a_job_fails():
    parent = _Parent()
    with pytest.raises(RuntimeError):
        _dispatch([_Job(0), _Job(1, A="fail"), _Job(2)], parent)
    assert parent.created
    assert all(r.resets == [r.thread] for r in parent.created)
