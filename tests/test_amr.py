def _post_in_waveguide():
    import emerge as em

    em.cleanup()
    model = em.Simulation("AMR", loglevel="INFO")
    a, b, L = 0.02286, 0.01016, 0.04
    box = em.geo.Box(a, L, b)
    post = em.geo.Cylinder(0.002, b, em.ZAX.construct_cs((a / 3, L / 2, 0)))
    wg = em.geo.subtract(box, post)
    model.commit_geometry()
    model.mw.set_frequency_range(9e9, 11e9, 3)
    model.mw.set_resolution(0.2)
    model.generate_mesh()
    p1 = model.mw.bc.ModalPort(wg.face("front", tool=box), 1)
    p2 = model.mw.bc.ModalPort(wg.face("back", tool=box), 2)
    p1.align_modes(em.ZAX)
    p2.align_modes(em.ZAX)
    return model


def test_amr_local_refinement():
    """Three local refinement passes: the mesh grows, stays valid and the S-parameters settle."""
    import emerge as em
    import numpy as np

    model = _post_in_waveguide()
    n0 = model.mesh.n_tets
    model.adaptive_mesh_refinement(max_steps=3, frequency=10e9, convergence=1e-9)
    n1 = model.mesh.n_tets
    assert n1 > 1.3 * n0, (n0, n1)
    model.mesh.diagnose()

    data = model.mw.run_sweep(False)
    S21 = np.abs(data.scalar.grid.S(2, 1))
    assert np.all(S21 > 0.05) and np.all(S21 < 1.0 + 1e-6)
    em.cleanup()


def test_amr_thin_conductor():
    """AMR on a finite conductivity (ThinConductor) trace: the refined basis needs its DoF split too."""
    import emerge as em
    import numpy as np

    em.cleanup()
    mm = 0.001
    model = em.Simulation("AMRThin", loglevel="ERROR")
    pcb = em.geo.PCB(1.6, unit=mm, material=em.Material(er=4.2), trace_material=em.Material(cond=5.8e7))
    pcb.new(0, 0, 3.0, (1, 0)).store("p1").straight(20).store("p2")
    pcb.compile_paths(merge=True)
    pcb.determine_bounds(topmargin=10, bottommargin=10)
    pcb.generate_pcb()
    pcb.generate_air(5)
    pcb.modal_port(pcb["p1"], 1, width_multiplier=5, height=5)
    pcb.modal_port(pcb["p2"], 2, width_multiplier=5, height=5)
    model.commit_geometry()
    model.mw.set_frequency_range(2e9, 3e9, 2)
    model.mw.set_resolution(0.3)
    model.generate_mesh()

    n0 = model.mesh.n_tets
    model.adaptive_mesh_refinement(max_steps=2, frequency=3e9, convergence=1e-9)
    from emerge._emerge.physics.microwave.bcs import ThinConductor
    assert len(model.mw.bc.oftype(ThinConductor)) > 0
    assert model.mesh.n_tets > n0

    S21 = np.abs(model.mw.run_sweep(False).scalar.grid.S(2, 1))
    assert np.all(S21 > 0.9) and np.all(S21 < 1.0 + 1e-6)
    em.cleanup()


def test_amr_periodic():
    """AMR on a periodic unit cell: paired faces are refined identically and power is conserved."""
    import emerge as em
    import numpy as np
    from emerge._emerge.aura.emerge_tools import periodic_report

    mm = 1e-3
    em.cleanup()
    model = em.Simulation("AMRPeriodic", loglevel="ERROR")
    cell = em.RectCell(10 * mm, 10 * mm)
    model.set_periodic_cell(cell)
    cell.volume(0, 30 * mm)
    em.geo.Sphere(3 * mm, position=(0, 0, 15 * mm)).set_material(em.Material(6))
    top, bot = cell.port_face(30 * mm), cell.port_face(0)
    cell.set_scanangle(30, 20)
    model.commit_geometry()
    model.mw.set_frequency(10e9)
    model.mw.set_resolution(0.2)
    model.generate_mesh()
    model.mw.bc.floquet_port(top, 1)
    model.mw.bc.floquet_port(bot, 2)

    periodic = [(bc.face1.tags, bc.face2.tags, np.asarray(bc.dv)) for bc in model.mesher._get_periodic_bcs()]
    tris_before = [r["n_tris"][0] for r in periodic_report(model.mesh, periodic)]
    n0 = model.mesh.n_tets
    model.adaptive_mesh_refinement(max_steps=3, frequency=10e9, convergence=1e-9)
    assert model.mesh.n_tets > 1.3 * n0

    reports = periodic_report(model.mesh, periodic)
    assert sum(r["n_tris"][0] for r in reports) > sum(tris_before), "periodic faces were not refined"
    for r in reports:
        assert r["n_tris"][0] == r["n_tris"][1]
        assert r["unmatched"] == 0
        assert r["order_preserving"]

    model.mw.run_sweep()
    powers = model.mw._check_port_powers()
    assert abs(abs(powers[1]) - 1.0) < 1e-4
    assert abs(abs(powers[2]) - 1.0) < 1e-4
    em.cleanup()
