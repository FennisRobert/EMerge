from emerge.solvers import fem
import numpy as np

mm = 0.001
wga = 22.86*mm

L = 100*mm

dist = 5*mm
th = 1*mm
ap = 15*mm
N = 6
with fem.Simulation2D('NewModel') as model:
    wgtop = fem.geo2d.rectangle((-L/2,0), (L/2, wga/2))
    

    iris = fem.geo2d.rectangle((-N*dist/2-th/2, ap/2),(-N*dist/2+th/2,wga/2))
    for iris in fem.geo2d.array(iris, (dist,0), N):
        wgtop = wgtop.difference(iris)
    
    wgbot = fem.geo2d.mirror(wgtop, (0,0), (0,1))

    wg = wgtop.union(wgbot)
    #wg = wg.simplify(1e-6)


    model.define_geometry([wg,])

    model.geo.overview()

    ep1 = model.geo.select(-L/2,0)
    ep2 = model.geo.select(L/2,0)

    edges1 = model.geo.edges.get_edges([57,58])
    edges2 = model.geo.edges.get_edges([26,27])
    model.physics.assign(fem.bc.Port(1, True), edges=edges1)#edge = ep1.edge)
    model.physics.assign(fem.bc.Port(2), edges=edges2)#edge=ep2.edge)

    model.physics.set_frequency(10e9)

    sol = model.run_frequency_domain()

    fem.plot.plot_field(sol, np.abs(sol.Ez[0,:]))

    print(sol.S.shape)
    print(20*np.log10(np.abs(sol.S[0,:,:])))