from emerge.solvers import fem
import numpy as np
from emerge.plot import eplot, Line

mm = 0.001
wga = 22.86*mm
Lfeed = 100*mm

Hair = 100*mm
Wair = 50*mm
abc_order = 2
x0 = -Lfeed/2
y0 = Wair/2
th = 2*mm
#abcfunc = None
with fem.Simulation2D("MyModel") as model:
    #waveguide = fem.geo.rectangle((-Lfeed,-wga/2), (0,wga/2))
    waveguide = fem.geo2d.polygon([(-Lfeed,-wga/2),(-Lfeed/2,-wga/2),(-th,-1.2*wga),(-th,1.2*wga),(-Lfeed/2,wga/2),(-Lfeed,wga/2)])
    
    slit = fem.geo2d.rectangle((-2*th,-1*wga),(th,1*wga))
    air, boundary = fem.geo2d.circle((0,0), Hair/2, 21, ang_range=[-90,90])

    model.define_geometry(fem.geo2d.rasterize([waveguide, air, slit], 1e-6))
    
    model.geo.overview()

    model.physics.assign(fem.bc.Port(port_number=1, active=True),edge=model.geo.select(-Lfeed,0).edge)
    
    sphere_edges = model.geo.on_boundary(boundary)
    abc = model.physics.assign(fem.bc.ABC.spherical(order=abc_order, origin=(200*mm, 300*mm)), edges=sphere_edges)

    model.resolution = 0.3

    model.physics.set_frequency(10e9)

    mesh = model.generate_mesh()
    
    
    model.run_frequency_domain()
    #for bc in model.physics.boundary_conditions:
    #    mesh.plot_mesh(highlight_vertices=bc.all_vertices)

    #fem.plot.plot_s_parameters(model.physics.solution)

    sol = model.physics.solution
    N = 16
    angs = np.linspace(-np.pi,np.pi,1000)
    farfield = fem.post.compute_farfield(angs, sol.Ez[0,:].squeeze(), sol.Hx[0,:].squeeze(), sol.Hy[0,:].squeeze(), mesh, [b.tag for b in sphere_edges], 2*np.pi*sol.freqs[0]/299792458)
    
    eplot([Line(angs*180/np.pi, np.abs(farfield), name='2D Farfield'),], xlabel='Angle (deg)',ylabel='Ez Farfield (V/m)')

    fem.plot.plot_field(sol, np.abs(sol.Ez[0,:]))
    fem.plot.animate_field(sol, sol.Ez[0,:], 36)

    print(20*np.log10(np.abs(sol.S)), sol.S, np.abs(sol.S))
    