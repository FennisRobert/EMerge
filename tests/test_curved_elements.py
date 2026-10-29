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


def _post_in_box(top_bc: str):
    em.cleanup()
    m = em.Simulation("CurvedRim", loglevel="ERROR")
    m.settings.mw_curved_elements = True
    a, b, L = 0.02286, 0.01016, 0.04
    box = em.geo.Box(a, L, b)
    post = em.geo.Cylinder(0.004, b, em.ZAX.construct_cs((a / 2, L / 2, 0)))
    wg = em.geo.subtract(box, post)
    m.commit_geometry()
    m.mw.set_frequency(10e9)
    m.mw.set_resolution(0.2)
    m.generate_mesh()
    top = wg.face("top", tool=box)
    if top_bc == "abc":
        m.mw.bc.AbsorbingBoundary(top)
    else:
        m.mw.bc.ModalPort(top, 1)
    return m


@pytest.mark.filterwarnings("ignore")
def test_absorbing_boundary_keeps_touching_curved_faces_curved():
    # the post ends on the absorbing top face: only the edges on that face stay straight, the post is curved
    from emerge._emerge.physics.microwave.assembly.curvilinear import curved_data_bcs, exclude_tags

    m = _post_in_box("abc")
    bcs = m.mw.bc.boundary_conditions
    cd = curved_data_bcs(m.mesh, bcs)
    assert cd is not None and len(cd[0]) > 10
    tets, edges = np.asarray(m.mesh.tets), np.asarray(m.mesh.edges)
    keys = {tuple(sorted(edges[:, e])) for t in exclude_tags(bcs) for e in m.mesh.ftag_to_edge[t]}
    from emerge._emerge.physics.microwave.assembly.curvilinear import _PAIRS
    for t, d in zip(*cd):
        for k, (i, j) in enumerate(_PAIRS):
            if tuple(sorted((tets[i, t], tets[j, t]))) in keys:
                assert not np.any(d[k])
    em.cleanup()


@pytest.mark.filterwarnings("ignore")
def test_wave_port_keeps_touching_curved_faces_straight():
    from emerge._emerge.physics.microwave.assembly.curvilinear import curved_data_bcs

    m = _post_in_box("port")
    assert curved_data_bcs(m.mesh, m.mw.bc.boundary_conditions) is None
    em.cleanup()
