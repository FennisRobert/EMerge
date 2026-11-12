# [Written by Claude] All code in this module was written by Claude (AI).
"""Which volume conductors the 3D assembler makes PEC: only those above settings.mw_3d_peclim (issue #94).
mw_2dbc_peclim is the limit for 2D surfaces and must not turn a lossy volume film into PEC."""
import emerge as em

mm = 0.001


def _film_strip(sigma: float) -> tuple[int, list[str]]:
    """A 2 x 10 x 0.2 mm film of conductivity sigma, 1 mm over a PEC ground, fed by a lumped port and shorted at
    the far end. The film is solved inside the volume: mw_3d_surfimplim and mw_3d_peclim are raised to 3e8 and
    3.1e8 S/m, while mw_2dbc_peclim keeps its default of 1e8 S/m.

    Returns the number of solved DoFs (all but the PEC DoFs) and the debug report of the conducting domains."""
    em.cleanup()
    sim = em.Simulation("ConductorLimits", loglevel="ERROR", reset_settings=True)
    sim.settings.mw_3d_surfimplim = 3.0e8
    sim.settings.mw_3d_peclim = 3.1e8
    film = em.Material(cond=sigma, er=1, name="film", _metal=True)
    em.geo.Box(10 * mm, 16 * mm, 6 * mm, position=(-4 * mm, -3 * mm, -3 * mm)).set_material(em.lib.AIR).background()
    em.geo.Plate((-2 * mm, -2 * mm, 0), (6 * mm, 0, 0), (0, 14 * mm, 0)).set_material(em.lib.PEC)
    em.geo.Plate((0, 10 * mm, 0), (2 * mm, 0, 0), (0, 0, 1 * mm)).set_material(em.lib.PEC)
    port = em.geo.Plate((0, 0, 0), (2 * mm, 0, 0), (0, 0, 1 * mm))
    em.geo.Box(2 * mm, 10 * mm, 0.2 * mm, position=(0, 0, 1 * mm)).set_material(film)
    sim.commit_geometry()
    sim.mw.set_frequency([1e9])
    sim.mw.set_resolution(0.2)
    sim.generate_mesh()
    sim.mw.bc.LumpedPort(port, 1, width=2 * mm, height=1 * mm, direction=em.ZAX)
    sim.mw.run_sweep()
    job, _ = sim.mw.assembler.assemble_freq_matrix(sim.mw.basis, sim.mw.mat_assy, sim.mw.bc.boundary_conditions, 1e9)
    result = len(job.solve_ids), sim.mw.assembler._conductor_report
    em.cleanup()
    return result


def test_film_above_2dbc_peclim_is_not_pec():
    # Below mw_2dbc_peclim, between it and mw_3d_peclim, above mw_3d_peclim
    (lossy, _), (film_2e8, report_2e8), (pec, report_pec) = (_film_strip(sigma) for sigma in (9e7, 2e8, 4e8))
    assert film_2e8 == lossy, "a film below mw_3d_peclim was made PEC"
    assert pec < lossy, "a film above mw_3d_peclim was not made PEC"
    assert "lossy volume" in report_2e8[0] and "PEC volume" in report_pec[0], (report_2e8, report_pec)
