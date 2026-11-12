"""Issue #98: a lossless TEM line between two ports conserves power with the second-order complete basis.

The cubic gradient DoFs (edge 3, face 6) of the complete sets are pinned on port faces. Without that, the 3D field
had a gradient part on the port face that the port absorbed but the S-parameters did not count (~12 % lost).
"""
import numpy as np
import pytest

import emerge as em

mm = 1e-3


def _power(port_type: str) -> np.ndarray:
    m = em.Simulation("port_complete", loglevel="ERROR")
    m.mw.set_basis_space(element_order=2, complete=True)
    air = em.geo.Box(1 * mm, 2 * mm, 1 * mm, position=(0, -1 * mm, -0.5 * mm)).set_material(em.lib.AIR)
    strip = em.geo.Plate((0, -0.15 * mm, 0), (1 * mm, 0, 0), (0, 0.3 * mm, 0))
    m.mw.set_frequency([5e9, 20e9])
    m.commit_geometry()
    m.mesher.max_size = 0.15 * mm
    m.mesher.min_size = 0.01 * mm
    m.mesher.set_face_size(strip, 0.05 * mm)
    m.generate_mesh()
    m.mw.bc.PEC(strip)
    port = m.mw.bc.ModalPort if port_type == "modal" else m.mw.bc.WavePort
    for n, face in ((1, air.left), (2, air.right)):
        port(face, n, modetype="TEM")
    g = m.mw.run_sweep().scalar.grid
    m._saved = True
    return np.abs(np.asarray(g.S(1, 1))) ** 2 + np.abs(np.asarray(g.S(2, 1))) ** 2


@pytest.mark.parametrize("port_type", ["modal", "wave"])
def test_complete_basis_conserves_power(port_type):
    assert np.allclose(_power(port_type), 1.0, atol=1e-3)
