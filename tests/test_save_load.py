import emerge as em


def test_joblib():

    with em.Simulation("savetest", save_file=True) as m:
        wg = em.geo.Box(0.023, 0.05, 0.01)
        m.commit_geometry()
        m.mw.set_frequency_range(8e9, 10e9, 21)
        m.mw.bc.RectangularWaveguide(wg.front, 1)
        m.mw.bc.RectangularWaveguide(wg.back, 2)
        m.generate_mesh()
        data = m.mw.run_sweep()

    em.cleanup()

    with em.Simulation("savetest", load_file=True) as m:
        data = m.data.mw
        g = data.scalar.grid

    em.cleanup()


def test_msgpack():

    with em.Simulation("savetest", save_file=True, store_system="msgpack") as m:
        wg = em.geo.Box(0.023, 0.05, 0.01)
        m.commit_geometry()
        m.mw.set_frequency_range(8e9, 10e9, 21)
        m.mw.bc.RectangularWaveguide(wg.front, 1)
        m.mw.bc.RectangularWaveguide(wg.back, 2)
        m.generate_mesh()
        data = m.mw.run_sweep()

    em.cleanup()

    with em.Simulation("savetest", load_file=True, store_system="msgpack") as m:
        data = m.data.mw
        g = data.scalar.grid

    em.cleanup()


if __name__ == "__main__":
    test_joblib()
    test_msgpack()
