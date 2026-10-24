# [Written by Claude]
"""A ThinConductor splits the DoFs on its sheet: the tets on one side keep the original DoFs, the tets on
the other side use copies. Every tet on the copied side that touches a split edge must use the copies, the
side must be the same for every sheet triangle, and the system must be assembled with that numbering."""
import numpy as np


def _microstrip():
    import emerge as em

    em.cleanup()
    mm = 0.001
    model = em.Simulation("ThinSplit", loglevel="ERROR")
    pcb = em.geo.PCB(1.6, unit=mm, material=em.Material(er=4.2), trace_material=em.Material(cond=5.8e7))
    pcb.new(0, 0, 3.0, (1, 0)).store("p1").straight(20).store("p2")
    pcb.compile_paths(merge=True)
    pcb.determine_bounds(topmargin=10, bottommargin=10)
    pcb.generate_pcb()
    pcb.generate_air(5)
    pcb.modal_port(pcb["p1"], 1, width_multiplier=5, height=5)
    pcb.modal_port(pcb["p2"], 2, width_multiplier=5, height=5)
    model.commit_geometry()
    model.mw.set_frequency(3e9)
    model.mw.set_resolution(0.3)
    model.generate_mesh()
    return model


def test_thin_conductor_split_sides():
    import emerge as em

    model = _microstrip()
    data = model.mw.run_sweep(False)
    basis = model.mw.basis
    mesh = basis.mesh
    t2f = basis.tet_to_field
    copy_of = basis._dof_mapping
    original_of = {v: k for k, v in copy_of.items()}
    assert len(copy_of) > 0

    # Every copy is used by the (assembly) numbering
    assert np.all(np.isin(list(original_of), t2f))

    # No tet uses a DoF and its copy at the same time
    for itet in range(t2f.shape[1]):
        ids = set(t2f[:, itet].tolist())
        assert not any(original_of.get(i) in ids for i in ids), itet

    # All tets with copies lie on the same side of the (flat) trace sheet, the others on the other side
    z_sheet = mesh.nodes[2, mesh.tris[:, np.array(sorted(basis._split_tris))]].mean()
    uses_copy = np.isin(t2f, list(original_of)).any(axis=0)
    uses_orig = np.isin(t2f, list(copy_of)).any(axis=0)
    zc = mesh.centers[2]
    assert np.all(zc[uses_copy] > z_sheet) or np.all(zc[uses_copy] < z_sheet)
    assert np.all(np.sign(zc[uses_orig] - z_sheet) != np.sign(zc[uses_copy][0] - z_sheet))

    assert np.abs(data.scalar.grid.S(2, 1))[0] > 0.9
    em.cleanup()
