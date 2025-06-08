from emerge.solvers import fem
import numpy as np
from scipy.optimize import differential_evolution
mm = 0.001
f0 = 10e9

wga = 22.86*mm
FeedL = 50*mm

optimizations = []
def compute_S11(coeff):
    HornL = coeff[0]
    ApertureW = coeff[1]

    print(str(HornL) + ' ' + str(ApertureW))
    S11 = 0
    with fem.Simulation2D('HornOptimization') as model:

        feed = fem.geo2d.rectangle((-FeedL-HornL,-wga/2),(-HornL+1*mm,wga/2))
        horn = fem.geo2d.polygon([(-HornL,-wga/2),(0,-ApertureW/2),(0,ApertureW/2),(-HornL,wga/2)])
        air, boundary = fem.geo2d.circle((-0.0001,0),70*mm, 31,[-90,90])
        
        
        
        model.define_geometry([feed.union(horn).union(air)])

        edges = model.geo.on_boundary(boundary)

        model.physics.assign(fem.bc.ABC.spherical(2,(0,0)), edges=edges)

        model.physics.assign(fem.bc.Port(1,True), edge=model.geo.select(-FeedL-HornL,0).edge)

        model.resolution = 0.28

        model.physics.set_frequency(9e9)

        mesh = model.generate_mesh()

        #mesh.plot_mesh()
        
        sol = model.run_frequency_domain()

        fem.plot.plot_field(sol, np.abs(sol.Ez[0,:]))
        S11 = np.abs(sol.S[0,0,0])
        print(f'Length = {HornL/mm:.1f}mm, Aperture = {ApertureW/mm:.1f}mm, S11dB = {S11:.1f}dB')
        if S11 > 1:
            fem.plot.plot_field(sol, np.abs(sol.Ez[0,:]))
    optimizations.append((coeff, S11))
    return S11

compute_S11([110.5*mm, 39.1*mm])

print(differential_evolution(compute_S11, bounds=[(10*mm,120*mm),(22.86*mm,100*mm)]))
for coeff, S11 in optimizations:
    print(f'Length = {coeff[0]/mm:.1f}mm, Aperture = {coeff[1]/mm:.1f}mm, S11dB = {S11:.5f}dB')