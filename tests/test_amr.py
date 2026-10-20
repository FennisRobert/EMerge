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
