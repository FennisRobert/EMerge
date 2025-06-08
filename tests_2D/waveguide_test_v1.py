from emerge.solvers import fem
import numpy as np

mm = 0.001

wga = 22.86*mm
L = 150*mm

th = 1*mm
ropen = 0.6

wiris = (1-ropen)*wga/2
N = 3

dist = 22*mm
y0 = 30*mm

with fem.Simulation2D("MyModel") as model:
    waveguide = fem.geo2d.rectangle((0, 0), (L, wga))

    for i in range(N):
        y1 = y0 + i*(dist+th)
        Liris = fem.geo2d.rectangle((y1,0),(y1+th,wiris))
        Riris = fem.geo2d.rectangle((y1,wga-wiris),(y1+th,wga))
        waveguide = waveguide.difference(Liris)
        waveguide = waveguide.difference(Riris)

    model.define_geometry([waveguide])
    
    model.geo.overview()
    
    model.physics.assign(model.geo.select(0,0.5*wga).edge, fem.bc.Port(active=True))
    model.physics.assign(model.geo.select(L,0.5*wga).edge, fem.bc.Port(active=False))

    model.resolution = 0.15

    model.physics.set_frequency(np.linspace(8e9,10e9,51))
    mesh = model.generate_mesh()

    mesh.plot_mesh()
    
    model.run_frequency_domain()
    
    fem.plot.plot_s_parameters(model.physics.solution)

    for sol in model.physics.solution:
    
        fem.plot.plot_field(sol, np.abs(sol.Ez))