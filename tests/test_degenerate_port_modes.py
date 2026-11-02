# [Written by Claude] All code in this module was written by Claude (AI).
import numpy as np

import emerge as em
from emerge._emerge.physics.microwave.bcs.mode_gauge import port_face_quadrature

""" Ports with several modes of the same propagation constant."""

mm = 0.001


def _cross_powers(sim, port, k0) -> np.ndarray:
    """The matrix 1/2 ∬ (E_i × H_j*) · n dS of the stored port modes."""
    quad = port_face_quadrature(sim.mesh.nodes, sim.mesh.tris[:, sim.mesh.get_triangles(port.tags)], port.cs.zax.np)
    modes = port.get_modes(k0)
    E = [mode.E_function(*quad.points) for mode in modes]
    H = [mode.H_function(*quad.points) for mode in modes]
    n = port.cs.zax.np
    return np.array([[0.5 * np.sum(n @ np.cross(Ei, np.conj(Hj), axis=0) * quad.weights) for Hj in H] for Ei in E])


def test_degenerate_modes_are_orthogonal():
    # Two PEC wires in an air-filled PEC box: two TEM modes with the same propagation constant at each port.
    sim = em.Simulation("twowire")
    box = em.geo.Box(10 * mm, 10 * mm, 20 * mm, position=(-5 * mm, -5 * mm, 0))
    w1 = em.geo.Cylinder(0.6 * mm, 20 * mm, cs=em.GCS.displace(-2 * mm, 0, 0))
    w2 = em.geo.Cylinder(0.6 * mm, 20 * mm, cs=em.GCS.displace(2 * mm, 0, 0))
    air = em.geo.subtract(em.geo.subtract(box, w1), w2)
    sim.commit_geometry()
    sim.mw.set_frequency(3e9)
    sim.generate_mesh()
    p1 = sim.mw.bc.ModalPort(air.face("-z"), 1, number_of_modes=2)
    p2 = sim.mw.bc.ModalPort(air.face("+z"), 2, number_of_modes=2)
    data = sim.mw.run_sweep()

    k0 = 2 * np.pi * 3e9 / 299792458
    for port in (p1, p2):
        P = np.abs(_cross_powers(sim, port, k0))
        assert P.shape == (2, 2)
        assert np.allclose(np.diag(P), 1.0, atol=1e-6)
        assert P[0, 1] < 1e-6 and P[1, 0] < 1e-6

    # The model is lossless: S is unitary.
    S = data.scalar.grid.Sp[0]
    assert np.abs(S.conj().T @ S - np.eye(4)).max() < 1e-3
    em.cleanup()


def test_complex_combinations_of_real_modes_get_a_real_basis():
    # The eigen solver may return complex combinations of two real modes; no phase factor makes them real.
    from emerge._emerge.physics.microwave.microwave_3d import Microwave3D
    rng = np.random.default_rng(1)
    R = rng.standard_normal((50, 2))
    V = R @ np.array([[1.0, 0.3 + 0.8j], [0.5j, 1.0 - 0.2j]])
    B = Microwave3D._real_basis(V)
    assert np.abs(B.imag).max() == 0.0
    # same span as R
    assert np.linalg.norm(R - B @ np.linalg.lstsq(B.real, R, rcond=None)[0]) < 1e-10
    # a lossy eigenspace (complex vectors of rank 2 * 2 in real and imaginary part) stays as it is
    C = rng.standard_normal((50, 2)) + 1j * rng.standard_normal((50, 2))
    assert np.array_equal(Microwave3D._real_basis(C), C)


if __name__ == "__main__":
    test_degenerate_modes_are_orthogonal()


def test_iterative_modal_analysis_finds_full_cluster():
    # Round guide (R = 14.91 mm) at 15 GHz: TE01 and the TM11 pair share β = 180.5 rad/m (p'01 = p11). The
    # iterative solver used to drop both TM11 modes (their x^H B x of a unit vector is below 0.1) and return the
    # TE31 pair instead.
    em.cleanup()
    with em.Simulation("roundguide", loglevel="ERROR") as sim:
        cyl = em.geo.Cylinder(14.91 * mm, 30 * mm)
        sim.commit_geometry()
        sim.mw.set_frequency([15e9])
        sim.mw.set_resolution(0.10)
        sim.generate_mesh()
        port = sim.mw.bc.ModalPort(cyl.front, 1, number_of_modes=8)
        sim.mw.modal_analysis(port, 8, direct=False, freq=15e9, target_kz=289.0)
        k0 = 2 * np.pi * 15e9 / 299792458
        betas = np.sort([np.real(mode.beta) for mode in port.get_modes(k0)])[::-1]
    np.testing.assert_allclose(betas, [289.0, 289.0, 269.7, 238.2, 238.2, 180.5, 180.5, 180.5], atol=0.1)


def test_modal_port_too_few_propagating_modes():
    # WR-90 at 10 GHz has one propagating mode. A ModalPort with two modes stops with a clear error before the
    # sweep (it used to end in a ZeroDivisionError in the power normalization after it).
    import pytest
    from emerge._emerge.physics.microwave.bcs.port_bcs import BoundaryConditionError

    em.cleanup()
    with em.Simulation("wr90", loglevel="ERROR") as sim:
        box = em.geo.Box(22.86 * mm, 10.16 * mm, 30 * mm)
        sim.commit_geometry()
        sim.mw.set_frequency([10e9])
        sim.generate_mesh()
        sim.mw.bc.ModalPort(box.face("-z"), 1, number_of_modes=2)
        sim.mw.bc.ModalPort(box.face("+z"), 2, number_of_modes=2)
        with pytest.raises(BoundaryConditionError, match="1 propagating mode"):
            sim.mw.run_sweep()
