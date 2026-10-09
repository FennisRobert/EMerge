
def test() -> None:
    import emerge as em
    import numpy as np
    from emerge.plot import smith, plot_sp

    """ STEPPED IMPEDANCE FILTER

    In this demo we will look at how we can construct a stepped impedance filter using the
    PCB Layouter interface in EMerge.

    """
    # First we will define some constants/variables/parameters for our simulation.
    mm = 0.001
    mil = 0.0254*mm

    L0, L1, L2, L3 = 400, 660, 660, 660 # The lengths of the sections in mil's
    W0, W1, W2, W3 = 50, 128, 8, 224 # The widths of the sections in mil's

    th = 62 # The PCB Thickness
    pcbmat = em.Material(er=2.2, tand=0.00, color="#217627")

    # We start by creating our simulation object.

    m = em.Simulation('Demo1_SIF', loglevel='TRACE', write_log=True)

    layouter = em.geo.PCB(th, unit=mil, material=pcbmat, layers=3)

    layouter.new(0,0,W0, (1,0), 1).store('p1').straight(L0, W0).straight(L1,W1).straight(L2,W2).straight(L3,W3)\
        .straight(L2,W2).straight(L1,W1).straight(L0,W0).store('p2')

    p1 = layouter.modal_port(layouter.load('p1'), 1, height=0)
    p2 = layouter.modal_port(layouter.load('p2'), 2, height=0)
    polies = layouter.compile_paths(merge=True)

    layouter.determine_bounds(leftmargin=0, topmargin=200, rightmargin=0, bottommargin=200)

    pcb = layouter.generate_pcb(True, merge=True)

    m.commit_geometry()

    m.mw.set_resolution(0.08)

    m.mw.set_frequency_range(0.2e9, 8e9, 8)

    m.mesher.set_boundary_size(polies, 3*mm, growth_rate=1.2)
    # Finally we generate our mesh and view it
    m.generate_mesh()

    #m.view(use_gmsh=True)

    # Finally we execute the frequency domain sweep and compute the Scattering Parameters.
    sol = m.mw.run_sweep(parallel=True, n_workers=4, frequency_groups=8, multi_processing=False)
    
if __name__ == "__main__":
    test()