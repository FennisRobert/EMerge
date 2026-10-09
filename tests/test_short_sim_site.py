
import emerge as em
def test():
    
    em.cleanup()
    m = em.Simulation('rectwaveguide')
    box = em.geo.Box(0.02286, 0.05, 0.01016)
    m.mw.set_frequency_range(8e9, 10e9, 31)
    m.generate_mesh()
    m.mw.bc.RectangularWaveguide(box.face('front'), 1)
    m.mw.bc.RectangularWaveguide(box.face('back'), 2)
    data = m.mw.run_sweep()
    
    em.cleanup()
    
if __name__=="__main__":
    test()