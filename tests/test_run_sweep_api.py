# [Written by Claude] run_sweep / run_scattered are sequential; the *_distributed versions solve several frequency
# points at once. The old parallel=True, n_workers=... arguments still work for now, with a deprecation warning.
import pytest
from emerge._emerge.physics.microwave.microwave_3d import Microwave3D


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_driver(name, order):
        def driver(self, *args, _reset_solvers=True, **kwargs):
            opts = {"parallel": False, "n_workers": 2, "multi_processing": False, "frequency_groups": -1,
                    "nthreads": None}
            opts.update(zip(order, args))
            opts.update(kwargs)
            recorded.append((name, {k: opts[k] for k in ("parallel", "n_workers", "multi_processing",
                                                         "frequency_groups", "nthreads")}))
        return driver

    monkeypatch.setattr(Microwave3D, "_run_sweep", fake_driver("sweep", Microwave3D._LEGACY_SWEEP_ARGS))
    monkeypatch.setattr(Microwave3D, "_run_scattered", fake_driver("scattered", Microwave3D._LEGACY_SCATTERED_ARGS))
    return recorded


def _mw():
    return Microwave3D.__new__(Microwave3D)


def _opts(parallel=False, n_workers=2, multi_processing=False, frequency_groups=-1, nthreads=None):
    return dict(parallel=parallel, n_workers=n_workers, multi_processing=multi_processing,
                frequency_groups=frequency_groups, nthreads=nthreads)


@pytest.mark.parametrize("method,name", [("run_sweep", "sweep"), ("run_scattered", "scattered")])
def test_sequential(calls, method, name):
    getattr(_mw(), method)(nthreads=6, frequency_groups=4)
    assert calls == [(name, _opts(nthreads=6, frequency_groups=4))]


@pytest.mark.parametrize("method,name", [("run_sweep", "sweep"), ("run_scattered", "scattered")])
def test_distributed(calls, method, name):
    getattr(_mw(), method + "_distributed")(n_workers=3, nthreads=4)
    getattr(_mw(), method + "_distributed")(n_workers=2, worker_type="process")
    assert calls == [(name, _opts(True, 3, nthreads=4)), (name, _opts(True, 2, True))]
    with pytest.raises(ValueError, match="worker_type"):
        getattr(_mw(), method + "_distributed")(worker_type="threading")


@pytest.mark.parametrize("method,name", [("run_sweep", "sweep"), ("run_scattered", "scattered")])
def test_legacy_parallel_arguments_forward(calls, method, name):
    getattr(_mw(), method)(parallel=True, n_workers=4, frequency_groups=8, multi_processing=True)
    getattr(_mw(), method)(True, 3)
    assert calls == [(name, _opts(True, 4, True, 8)), (name, _opts(True, 3))]


def test_unknown_keyword_raises(calls):
    with pytest.raises(TypeError, match="njobs"):
        _mw().run_sweep(njobs=3)
