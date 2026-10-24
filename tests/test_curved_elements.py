# [Written by Claude] All code in this module was written by Claude (AI).
"""Curved (isoparametric) tetrahedra at curved CAD surfaces (Settings.mw_curved_elements)."""
import numpy as np
import pytest
from scipy.special import jn_zeros

import emerge as em
from emr.curlcurl import _matrix_builder
from emr.curlcurl_iso import _matrix_builder_iso
from emerge._emerge.physics.microwave.assembly.curvilinear import quadrature_rule

A, H = 10e-3, 8e-3                                   # cylinder radius and height, TM010 is the lowest mode
F010 = jn_zeros(0, 1)[0] * 299792458 / (2 * np.pi * A)


def _cavity(curved: bool, resolution: float):
    m = em.Simulation("CurvedCavity", loglevel="ERROR")
    m.settings.mw_curved_elements = curved
    em.geo.Cylinder(radius=A, height=H, cs=em.GCS)
    m.commit_geometry()
    m.mw.set_frequency(F010)
    m.mw.set_resolution(resolution)
    m.generate_mesh()
    data = m.mw.eigenmode(0.95 * F010, nmodes=3)
    f = min((abs(fd.freq.real) for fd in data.field._data_entries), key=lambda x: abs(x - F010))
    return m, (f - F010) / F010


@pytest.mark.filterwarnings("ignore")
def test_curved_cylinder_cavity_resonance():
    _, err_straight = _cavity(False, 0.2)
    _, err_curved = _cavity(True, 0.2)
    # the flat facets shift TM010 by about +1 %; on the same mesh the curved tets are within 0.05 %
    assert err_straight > 5e-3
    assert abs(err_curved) < 5e-4


@pytest.mark.filterwarnings("ignore")
def test_zero_displacement_gives_straight_matrices():
    m, _ = _cavity(False, 0.33)
    field = m.mw.basis
    mesh = field.mesh
    nt = mesh.tets.shape[1]
    rng = np.random.default_rng(1)
    er = np.zeros((3, 3, nt), np.complex128)
    ur = np.zeros((3, 3, nt), np.complex128)
    for t in range(nt):
        q = rng.standard_normal((3, 3))
        er[:, :, t] = np.eye(3) * (2 - 0.1j) + 0.2 * (q + q.T)
        ur[:, :, t] = np.eye(3) * 1.5 + 0.1 * (q @ q.T)
    ids = np.arange(0, nt, 3, dtype=np.int64)
    args = (mesh.nodes, mesh.tets, mesh.tris, mesh.edges)
    E0, B0, _, _ = _matrix_builder(*args, field.tet_to_field, mesh.tet_to_edge, mesh.tet_to_tri, ur, er, ids,
                                   field.dofcodes3d)
    E1, B1 = _matrix_builder_iso(*args, mesh.tet_to_edge, mesh.tet_to_tri, ur, er, ids,
                                 np.zeros((len(ids), 6, 3)), field.dofcodes3d, quadrature_rule(4))
    assert np.allclose(E1, E0, rtol=0, atol=1e-9 * np.abs(E0).max())
    assert np.allclose(B1, B0, rtol=0, atol=1e-9 * np.abs(B0).max())
