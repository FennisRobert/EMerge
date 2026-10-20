import numpy as np

from emerge._emerge.physics.microwave.microwave_data import MWScalarNdim, renormalise_s


def _dataset(Z0: np.ndarray, seed: int = 0) -> MWScalarNdim:
    """A random reciprocal 4-port with F=3 frequency points."""
    rng = np.random.default_rng(seed)
    F, N = 3, 4
    S = 0.3 * (rng.standard_normal((F, N, N)) + 1j * rng.standard_normal((F, N, N)))
    S = 0.5 * (S + np.transpose(S, (0, 2, 1)))
    d = MWScalarNdim()
    d.freq = np.linspace(1e9, 2e9, F)
    d.Sp = S
    d.Z0 = np.broadcast_to(np.asarray(Z0, dtype=complex), (F, N)).copy()
    d._portmap = {1: 0, 2: 1, 3: 2, 4: 3}
    d._portnumbers = [1, 2, 3, 4]
    return d


def _mixed_mode(S: np.ndarray) -> np.ndarray:
    """Reference mixed-mode conversion of ports (1,2) via the orthonormal transform M S M^T."""
    M = np.eye(4)
    M[:2, :2] = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    return M @ S @ M.T


def test_combine_ports_matches_mixed_mode_transform():
    d = _dataset(50.0)
    S0 = d.Sp.copy()
    d.combine_ports(1, 2)
    assert np.allclose(d.Sp, _mixed_mode(S0))
    assert np.allclose(d.Z0, 50.0)


def test_combine_ports_renormalizes_first():
    Z0 = np.array([48.0 + 2j, 52.0 - 1j, 50.0, 49.0])
    d = _dataset(Z0)
    S0 = d.Sp.copy()
    d.combine_ports(1, 2, Z0renorm=50.0)

    expected = _mixed_mode(renormalise_s(S0, np.tile(Z0, (3, 1)), 50.0))
    assert np.allclose(d.Sp, expected)
    # the new reference impedance is stored so later renormalization starts from it
    assert np.allclose(d.Z0, 50.0)


def test_combine_ports_renorm_to_same_impedance_is_noop():
    a = _dataset(50.0, seed=1)
    b = _dataset(50.0, seed=1)
    a.combine_ports(1, 2)
    b.combine_ports(1, 2, Z0renorm=50.0)
    assert np.allclose(a.Sp, b.Sp)
