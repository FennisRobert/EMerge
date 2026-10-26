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


if __name__ == "__main__":
    test_degenerate_modes_are_orthogonal()
