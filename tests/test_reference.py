def test_wg1():
    import emerge as em
    import numpy as np
    from emerge.plot import plot_sp

    em.cleanup()
    model = em.Simulation("Wg1", loglevel="INFO")

    a, b, L = 0.02286, 0.01016, 0.05
    box = em.geo.Box(a, L, b)
    model.commit_geometry()
    model.mw.set_frequency_range(8e9, 12e9, 11)
    model.mw.set_resolution(0.1)
    model.generate_mesh()

    p1 = model.mw.bc.ModalPort(box.face("front"), 1)
    p2 = model.mw.bc.ModalPort(box.face("back"), 2)
    p1.align_modes(em.ZAX)
    p2.align_modes(em.ZAX)
    model.set_solver(em.EMSolver.AASDS)
    data = model.mw.run_sweep(False)
    glob = data.scalar.grid

    k0 = 2 * np.pi * glob.freq / 299792458
    kz = np.sqrt(k0**2 - (np.pi / a) ** 2)
    S21 = np.exp(-1j * kz * L)

    #plot_sp(glob.freq, [glob.S(1,1), glob.S(2,1)], labels=['S11','S21'])
    assert np.all(20 * np.log10(np.abs(glob.S(2, 1).real - np.real(S21))) < -30)
    assert np.all(20 * np.log10(np.abs(glob.S(2, 1).imag - np.imag(S21))) < -30)

    em.cleanup()

def test_wg2():
    import emerge as em
    import numpy as np
    from emerge.plot import plot_sp

    em.cleanup()
    model = em.Simulation("Wg1", loglevel="INFO")

    a, b, L = 0.02286, 0.01016, 0.05
    box = em.geo.Box(a, L, b)
    model.commit_geometry()
    model.mw.set_frequency_range(8e9, 12e9, 11)
    model.mw.set_resolution(0.1)
    model.generate_mesh()

    p1 = model.mw.bc.RectangularWaveguide(box.face("front"), 1)
    p2 = model.mw.bc.RectangularWaveguide(box.face("back"), 2)
    p1.align_modes(em.ZAX)
    p2.align_modes(em.ZAX)
    model.set_solver(em.EMSolver.AASDS)
    data = model.mw.run_sweep(False)
    glob = data.scalar.grid

    k0 = 2 * np.pi * glob.freq / 299792458
    kz = np.sqrt(k0**2 - (np.pi / a) ** 2)
    S21 = np.exp(-1j * kz * L)

    #plot_sp(glob.freq, [glob.S(1,1), glob.S(2,1)], labels=['S11','S21'])
    assert np.all(20 * np.log10(np.abs(glob.S(2, 1).real - np.real(S21))) < -30)
    assert np.all(20 * np.log10(np.abs(glob.S(2, 1).imag - np.imag(S21))) < -30)

    em.cleanup()

if __name__ == "__main__":
    test_wg1()
