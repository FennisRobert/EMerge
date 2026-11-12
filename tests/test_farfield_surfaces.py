# [Written by Claude] The side of an integration surface the fields are evaluated on: outside a PEC scatterer (inside
# has no field; this gave a zero RCS) and on the air side of an air/PML interface.
import numpy as np
import emerge as em


def test_rcs_pec_sphere():
    model = em.Simulation("RCSPECSphere")
    sphere = em.geo.Sphere(radius=0.25).set_material(em.lib.PEC)
    air = em.geo.Sphere(radius=0.75)
    model.commit_geometry()
    model.mw.set_frequency(300e6)
    model.mw.set_resolution(0.25)
    model.mesher.set_face_size(sphere.boundary(), 0.1)
    model.generate_mesh()

    scat = model.mw.bc.ScatteredField(air.boundary(), definition="EA")
    scat.set_excitations(polarizations=90)
    scat.radius = 0.75
    field = model.mw.run_scattered().field[0]

    ff = field.farfield_2d((-1, 0, 0), em.ZAX, sphere.boundary(), ang_range=(0, 180), Npoints=19)
    rcs = np.abs(ff.RCS)
    assert np.all(np.isfinite(rcs)) and rcs.max() > 0
    # ka = π/2: the forward and backscattered RCS are both of the order of the geometric cross section πa²
    assert 0.1 < rcs.max() / (np.pi * 0.25**2) < 10
    em.cleanup()


def test_radiated_power_on_pml_interface():
    # A dipole in an air box with PML on all sides: the power through the air/PML interface is the power accepted
    # by the port. The fields are sampled on the air side; on the PML side the power was 46 % too high.
    from emerge._emerge.geo.shapes import Alignment

    mm = 1e-3
    model = em.Simulation("DipolePML")
    L, w, gap = 23 * mm, 2 * mm, 2 * mm
    em.geo.Box(w, w, L, (-w / 2, -w / 2, gap / 2)).set_material(em.lib.PEC)
    em.geo.Box(w, w, L, (-w / 2, -w / 2, -gap / 2 - L)).set_material(em.lib.PEC)
    port = em.geo.Plate(np.array([-w / 2, 0, -gap / 2]), np.array([w, 0, 0]), np.array([0, 0, gap]))
    air, *_ = em.geo.pmlbox(80 * mm, 80 * mm, 100 * mm, (0, 0, 0), alignment=Alignment.CENTER,
                            thickness=20 * mm, N_mesh_layers=4, sides="tblrfa")
    model.commit_geometry()
    model.mw.set_frequency(3e9)
    model.mw.set_resolution(0.2)
    model.mesher.set_face_size(port, 0.5 * mm)
    model.generate_mesh()
    model.mw.bc.LumpedPort(port, 1, width=w, height=gap, direction=em.ZAX, Z0=50)
    data = model.mw.run_sweep()

    accepted = 1 - abs(data.scalar.grid.S(1, 1).ravel()[0]) ** 2
    ff = data.field[0].farfield_2d((1, 0, 0), (0, 0, 1), air.boundary(), (-180, 180), 37)
    assert abs(ff.Ptot - accepted) < 0.02 * accepted
    em.cleanup()
