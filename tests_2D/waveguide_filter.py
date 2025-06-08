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
    waveguide = fem.geo2d.rectangle((-Ltot/2,-wga/2), (Ltot/2,wga/2))
    iris1 = fem.geo2d.rect_center((spacing/2,(wga-Liris1)/2),th,Liris1*1.001)
    iris2 = fem.geo2d.rect_center((spacing*1.42,(wga-Liris2)/2),th,Liris2*1.001)
    iris3 = fem.geo2d.rect_center((-spacing/2,(wga-Liris1)/2),th,Liris1*1.001)
    iris4 = fem.geo2d.rect_center((-spacing*1.42,(wga-Liris2)/2),th,Liris2*1.001)

    iris5 = fem.geo2d.rect_center((spacing/2,-(wga-Liris1)/2),th,Liris1*1.001)
    iris6 = fem.geo2d.rect_center((spacing*1.42,-(wga-Liris2)/2),th,Liris2*1.001)
    iris7 = fem.geo2d.rect_center((-spacing/2,-(wga-Liris1)/2),th,Liris1*1.001)
    iris8 = fem.geo2d.rect_center((-spacing*1.42,-(wga-Liris2)/2),th,Liris2*1.001)

    for obj in [iris1,iris2,iris3,iris4,iris5,iris6,iris7,iris8]:
        waveguide = waveguide.difference(obj)

    model.define_geometry([waveguide,])
    
    #model.geo.overview()
    
    model.physics.assign(fem.bc.Port(port_number=1, active=True), edge=model.geo.select(-Ltot/2,0).edge)
    model.physics.assign(fem.bc.Port(port_number=2, active=False), edge=model.geo.select(Ltot/2,0).edge)

    model.resolution = 0.3

    model.physics.set_frequency(np.linspace(8.5e9,11.5e9,31))
    mesh = model.generate_mesh()

    mesh.plot_mesh()
    
    model.run_frequency_domain()
    
    fem.plot.plot_s_parameters(model.physics.solution)

    sol = model.physics.solution
    for i in range(len(model.physics.frequencies)):
    
        fem.plot.plot_field(sol, np.abs(sol.Ez[i,:]))