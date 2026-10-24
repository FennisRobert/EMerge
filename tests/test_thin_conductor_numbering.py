# [Written by Claude]
"""A ThinConductor splits the DoFs at its sheet. The system matrix is assembled with the original numbering
(get_tet_to_field), so the fields, far fields and the error estimate must read the same numbering through
tet_to_field. Otherwise the tetrahedra at the sheet read the split copies and the fields there are wrong."""
import numpy as np


def test_thin_conductor_fields_use_the_assembly_numbering():
    import emerge as em

    em.cleanup()
    mm = 0.001
    model = em.Simulation("ThinNumbering", loglevel="ERROR")
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
    data = model.mw.run_sweep(False)

    basis = model.mw.basis
    assert not np.array_equal(basis.tri_to_field_os, basis.tri_to_field)  # the sheet DoFs were split
    assert np.array_equal(basis.tet_to_field, basis.get_tet_to_field())
    assert np.abs(data.scalar.grid.S(2, 1))[0] > 0.9
    em.cleanup()
