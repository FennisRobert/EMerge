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


def _signed_port_powers(sim) -> dict:
    """Signed power flux of every port mode through its port face (inward normal)."""
    import numpy as np
    from emerge._emerge.physics.microwave.port_functions import compute_port_power_flux

    k0 = 2 * np.pi * sim.mw.frequencies[0] / 299792458
    powers = {}
    for port, index, mode_nr in sim.mw.bc.iter_port_modes():
        def fE(x, y, z):
            return tuple(port.port_mode_3d_global(x, y, z, k0=k0, which="E", mode_nr=mode_nr))

        def fH(x, y, z):
            return tuple(port.port_mode_3d_global(x, y, z, k0=k0, which="H", mode_nr=mode_nr))

        tris = sim.mesh.get_triangles(port.tags)
        powers[index] = compute_port_power_flux(
            sim.mesh.nodes, sim.mesh.tris[:, tris], fE, fH, sim.mesh.inward_normal(port.tags)
        )
    return powers


def test_tm_mode_power_near_cutoff():
    """The TM01 mode of a circular guide between cutoff and sqrt(2) x cutoff carries +1 W into the model.

    The transverse H of TM modes depends on the sign convention of Ez; with an inconsistent
    sign the power flux of such a mode comes out negative.
    """
    sim = em.Simulation("tmpower")
    # r = 10 mm: TE11 cutoff 8.8 GHz, TM01 cutoff 11.5 GHz, next modes above 14.5 GHz
    guide = em.geo.Cylinder(0.01, 0.02, em.GCS)
    sim.commit_geometry()
    sim.mw.set_frequency(13e9)
    sim.mw.set_resolution(0.15)
    sim.generate_mesh()

    p1 = sim.mw.bc.ModalPort(guide.face("-z"), 1, modetype="TM")
    p2 = sim.mw.bc.ModalPort(guide.face("+z"), 2, modetype="TM")
    # Solve for TE11 (x2) and TM01 so the TM01 mode is found
    sim.mw.modal_analysis(p1, 4, freq=13e9)
    sim.mw.modal_analysis(p2, 4, freq=13e9)
    sim.mw.run_sweep()

    for index, power in _signed_port_powers(sim).items():
        assert abs(power - 1.0) < 1e-3, (index, power)
    em.cleanup()
