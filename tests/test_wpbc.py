def _microstrip(er: float, dense_ports: bool):
    import emerge as em

    mm = 0.001
    em.cleanup()
    sim = em.Simulation("wpbc", loglevel="INFO")
    pcb = em.geo.PCB(0.5 * mm, 1.0, material=em.Material(er=er, name="Dielectric"))
    w0 = pcb.calc.z0(50)
    pcb.new(0, 0, w0, (1, 0), 1)[1].straight(30 * mm)[2]
    path = pcb.compile_paths(True)
    pcb.determine_bounds(0, 7 * mm, 0, 7 * mm)
    mp1 = pcb.modal_port(1, 1, 4 * mm, mode_type="TEM")
    mp2 = pcb.modal_port(2, 2, 4 * mm, mode_type="TEM")
    pcb.generate_all(height_air_top=5 * mm)
    sim.commit_geometry()
    sim.mw.set_frequency_range(2e9, 6e9, 3)
    sim.mesher.set_boundary_size(path, 0.5 * mm)
    sim.generate_mesh()
    if dense_ports:
        sim.mw.bc.WavePortIH(mp1, 1)
        sim.mw.bc.WavePortIH(mp2, 2)
    data = sim.mw.run_sweep()
    grid = data.scalar.grid
    S11, S21 = grid.S(1, 1), grid.S(2, 1)
    em.cleanup()
    return S11, S21


def test_wave_port_quasi_tem_microstrip():
    """The dense wave port (WPBC) terminates a quasi-TEM microstrip (E_z != 0) without reflection."""
    import numpy as np

    S11, S21 = _microstrip(4.4, dense_ports=True)
    print("WPBC S11 dB", 20 * np.log10(np.abs(S11)), "|S21|", np.abs(S21))
    assert np.all(20 * np.log10(np.abs(S11)) < -25), S11
    assert np.all(np.abs(np.abs(S21) - 1) < 0.02), S21
