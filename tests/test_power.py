import emerge as em

""" Test if power normalization of ports works well."""


def test_coax_power():

    sim = em.Simulation("coaxpower")

    ro = 0.002
    ri = em.coax_rin(0.002, 1.0, 50)
    coax = em.geo.CoaxCylinder(ro, ri, 0.02, Nsections=20)

    sim.commit_geometry()
    sim.mw.set_frequency(5e9)
    sim.generate_mesh()

    p1 = sim.mw.bc.CoaxPort(coax.face("-z"), 1, rad_in_out=(ri, ro), er=1.0, cs=em.GCS)
    p2 = sim.mw.bc.ModalPort(coax.face("+z"), 2, modetype="TEM")

    data = sim.mw.run_sweep()
    powers = sim.mw._check_port_powers()
    
    assert abs(powers[1] - 1.0) < 1e-4
    assert abs(powers[2] - 1.0) < 1e-4
    em.cleanup()


def test_wg_power():

    sim = em.Simulation("wgpower")

    box = em.geo.Box(0.0228, 0.01, 0.01)

    sim.commit_geometry()
    sim.mw.set_frequency(10e9)
    sim.generate_mesh()

    p1 = sim.mw.bc.RectangularWaveguide(box.front, 1)
    p2 = sim.mw.bc.ModalPort(box.back, 2)

    data = sim.mw.run_sweep()
    powers = sim.mw._check_port_powers()
    
    assert abs(powers[1] - 1.0) < 1e-4
    assert abs(powers[2] - 1.0) < 1e-4
    em.cleanup()


def test_floquet_power():
    mm = 0.001
    sim = em.Simulation("fqtest")

    rectcell = em.RectCell(10 * mm, 10 * mm)
    sim.set_periodic_cell(rectcell)

    airvol = rectcell.volume(0, 50 * mm)

    fptop = rectcell.port_face(50 * mm)
    fpbot = rectcell.port_face(0 * mm)
    sim.commit_geometry()

    sim.mw.set_frequency(10e9)
    sim.mw.set_resolution(0.1)
    sim.generate_mesh()

    p1 = sim.mw.bc.floquet_port(fptop, 1)
    p2 = sim.mw.bc.floquet_port(fpbot, 2)
    data = sim.mw.run_sweep()
    powers = sim.mw._check_port_powers()
    assert abs(abs(powers[1]) - 1.0) < 1e-4
    assert abs(abs(powers[2]) - 1.0) < 1e-4
    em.cleanup()


if __name__ == "__main__":
    test_coax_power()
    test_wg_power()
    test_floquet_power()
