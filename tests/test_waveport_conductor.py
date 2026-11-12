# [Written by Claude] All code in this module was written by Claude (AI).
"""A WavePort whose face cuts a surface impedance conductor (the copper ground and trace of a microstrip port).
The port face DoFs inside the copper are touched by no assembled tet; they must stay PEC, otherwise the WavePort
term, which is zero inside the metal, leaves exactly zero matrix columns (SPARTA: "zero column (rank deficient)")."""
import numpy as np

import emerge as em

mm = 0.001


def test_waveport_through_copper_has_no_zero_columns():
    em.cleanup()
    sim = em.Simulation("WavePortCopper", loglevel="ERROR", reset_settings=True)
    t, h, w, L, W = 0.035, 0.5, 1.0, 6.0, 5.0
    sub = em.Material(er=3.5, name="sub")
    em.geo.Box(L * mm, W * mm, t * mm, position=(0, -W / 2 * mm, 0)).set_material(em.lib.COPPER)
    em.geo.Box(L * mm, W * mm, h * mm, position=(0, -W / 2 * mm, t * mm)).set_material(sub)
    em.geo.Box(L * mm, w * mm, t * mm, position=(0, -w / 2 * mm, (t + h) * mm)).set_material(em.lib.COPPER)
    em.geo.Box(L * mm, W * mm, 3 * mm, position=(0, -W / 2 * mm, 0)).set_material(em.lib.AIR).background()
    plates = [em.geo.Plate((x * mm, -W / 2 * mm, 0), (0, W * mm, 0), (0, 0, 3 * mm)) for x in (0, L)]
    sim.commit_geometry()
    sim.mw.set_frequency([10e9])
    sim.mw.set_resolution(0.2)
    sim.generate_mesh()
    for i, p in enumerate(plates):
        sim.mw.bc.WavePort(p, i + 1, modetype="TEM", number_of_modes=1)
    sim.mw.run_sweep()
    job, _ = sim.mw.assembler.assemble_freq_matrix(sim.mw.basis, sim.mw.mat_assy, sim.mw.bc.boundary_conditions, 10e9)
    A = job.A[job.solve_ids, :][:, job.solve_ids].tocsc()
    A.eliminate_zeros()
    empty = np.flatnonzero(np.diff(A.indptr) == 0)
    assert empty.size == 0, f"{empty.size} zero columns in the WavePort system"
    em.cleanup()
