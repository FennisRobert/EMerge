from emerge.solvers import fem

import numpy as np

Rlens = 0.75
Rdom = 1

f = 1
th = 0.005

wga = 0.2

with fem.Simulation2D('LensSim') as model:

    circle, boundary = fem.geo2d.circle((0,0), Rdom, 36)
    lens, boundary_lens = fem.geo2d.circle((0,0), Rlens, 25)
    #waveguide = fem.geo.rectangle((-Rdom-0.2,-wga/2),(-Rdom+0.1,wga/2))
    #circle = circle.union(waveguide)
    #reflector = fem.geo.rect_center((-Rlens-3*th,0), th, 0.5)
    #circle = circle.difference(reflector)

    point = fem.geo2d.shp.Point(-Rlens*f,0)

    model.define_geometry([circle,lens,point])

    lens_material = fem.material.Material(_fer = lambda x,y: (1/f)**2 * (1 + f**2 - (x**2+y**2)/Rlens**2))


    model.geo.get_domain(3).material = lens_material

    abc = model.physics.assign(fem.bc.ABC.spherical(2,(-Rlens,0)), edges = model.geo.on_boundary(boundary))

    model.physics.assign(fem.bc.PointSource(), point=model.geo.select(-Rlens,0).point)
    model.physics.set_frequency(1164e6)

    mesh = model.generate_mesh()
    model.update_boundary_conditions()

    mesh.plot_mesh(highlight_vertices=abc.all_nodes)

    sol = model.run_frequency_domain()

    angs = np.linspace(-np.pi,np.pi,1000)
    sphere_edges = model.geo.on_boundary(boundary)

    farfield = fem.post.compute_farfield(angs, sol.Ez[0,:].squeeze(), sol.Hx[0,:].squeeze(), sol.Hy[0,:].squeeze(), mesh, [b.tag for b in sphere_edges], 2*np.pi*sol.freqs[0]/299792458)
    
    fem.plot.eplot([fem.plot.Line(angs*180/np.pi, np.abs(farfield), name='2D Farfield'),], xlabel='Angle (deg)',ylabel='Ez Farfield (V/m)')

    fem.plot.radiation_plot(angs, np.abs(farfield))
    fem.plot.plot_field_tri(mesh, sol.er)
    fem.plot.plot_field(sol, np.real(sol.Ez[0,:]))

    fem.plot.animate_field(sol, sol.Ez[0,:], 35)

