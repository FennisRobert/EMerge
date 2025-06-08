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
    waveguide = fem.geo2d.rectangle((0,0), (wga/2,wga/2))
    
    model.define_geometry([waveguide, ])
    
    #model.geo.overview()
    
    
    model.resolution = 0.1

    model.physics.set_frequency(10e9)
    
    mesh = model.generate_mesh(element_order=1)

    #mesh.plot_mesh()
    
    sol = model.run_eigenmodes(nsolutions=2)

    for i in range(10):
        fem.plot.plot_field(sol, np.abs(sol.Ez[i,:]))

        fem.plot.animate_field(sol,sol.Ez[i,:], 35)

    