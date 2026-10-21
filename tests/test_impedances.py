def test_zchar_def():
    # Tests characteristic impedance values and if they are correct with numerical models.
    Z0_err_limit = 2.0

    import emerge as em

    sim = em.Simulation("Z0", loglevel="INFO")
    pcb = em.geo.PCB(1.0, 0.001, layers=3, material=em.lib.AIR)

    pcb.new(0, 0, 1.4423896, (0, 1), trace_layer=1)[1].straight(4.99654097)[2]

    trace = pcb.compile_paths(True)

    mp1 = pcb.modal_port(1, 1, 0, width=8)
    mp2 = pcb.modal_port(2, 2, 0, width=8)

    pcb.determine_bounds(5, 0, 5, 0)

    diel = pcb.generate_pcb()

    sim.commit_geometry()
    sim.mw.set_frequency(15e9)
    sim.mesher.set_boundary_size(trace, 0.00005)
    sim.generate_mesh()

    p1 = sim.mw.bc.get_port(1)
    p2 = sim.mw.bc.get_port(2)
    p1.set_integration_line((0, 0, -0.0005), (0, 0, 0), N=101)
    p2.set_integration_line((0, 0.00499654097, -0.0005), (0, 0.00499654097, 0), N=101)

    data = sim.mw.run_sweep()
    sim.display.populate()
    # sim.display.add_portmode(p1)
    # sim.display.add_portmode(p2)
    # sim.display.show()
    g = data.scalar.grid

    assert abs(g.Z0[0, 0] - 50) < Z0_err_limit and abs(g.Z0[0, 1] - 50) < Z0_err_limit
    sim.reset(physics=False, data=True, mesh=True)
    p1.reset()
    p2.reset()
    p1.impedance_definition = 'PI'
    p2.impedance_definition = 'PI'
    sim.generate_mesh()
    data = sim.mw.run_sweep()
    g = data.scalar.grid
    assert abs(g.Z0[0, 0] - 50) < Z0_err_limit and abs(g.Z0[0, 1] - 50) < Z0_err_limit

    sim.reset(physics=False, data=True, mesh=True)
    p1.reset()
    p2.reset()
    p1.impedance_definition = 'VI'
    p2.impedance_definition = 'VI'
    sim.generate_mesh()
    data = sim.mw.run_sweep()
    g = data.scalar.grid

    assert abs(g.Z0[0, 0] - 50) < Z0_err_limit and abs(g.Z0[0, 1] - 50) < Z0_err_limit
    em.cleanup()


if __name__ == "__main__":
    test_zchar_def()
