from emerge.solvers import fem
import numpy as np

mm = 0.001
cm = 0.01
wga = 2.286*cm
th = 0.3*cm
spacing = 2.03*cm
Liris1 = 0.695*cm
Liris2 = 0.53*cm
Ltot = 12*cm

with fem.Simulation2D("MyModel") as model:
    waveguide = fem.geo2d.rectangle((0,0), (Ltot/2,wga/2))

    iris1 = fem.geo2d.rect_center((spacing/2,(wga-Liris1)/2),th,Liris1*1.001)
    iris2 = fem.geo2d.rect_center((spacing*1.42,(wga-Liris2)/2),th,Liris2*1.001)
    
    waveguide = waveguide.difference(iris1)
    waveguide = waveguide.difference(iris2)

    waveguide = waveguide.union(fem.geo2d.mirror(waveguide, (0,0),(1,0)))
    waveguide = waveguide.union(fem.geo2d.mirror(waveguide,(0,0),(0,1))).simplify(1e-6)

    model.define_geometry([waveguide, ])
    
    model.geo.overview()
    
    model.physics.assign(fem.bc.Port(port_number=1, active=True), edge=model.geo.select(-Ltot/2,0).edge)
    model.physics.assign(fem.bc.Port(port_number=2, active=False), edge=model.geo.select(Ltot/2,0).edge)

    model.resolution = 0.3

    model.physics.set_frequency(np.linspace(8.5e9,11.5e9,21))
    
    mesh = model.generate_mesh()

    mesh.plot_mesh()
    
    sol = model.run_frequency_domain()
    
    fem.plot.plot_s_parameters(sol)

    fem.plot.plot_field(sol, np.abs(sol.Ez[11,:]))

    fem.plot.animate_field(sol, sol.Ez[11,:], 35)

    