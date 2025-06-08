from emerge.solvers import fem
import gmsh

mm = 0.001
wga = 22.86*mm
L = 100*mm
wgb = 10.16*mm

with fem.Simulation3D('MySimulation') as model:
    box1 = fem.geo3d.Box(L, wga, wgb)

    box2 = fem.geo3d.Box(5*mm, 5*mm, 20*mm, position=(L/2, wga/2, wgb/2), alignment=fem.geo3d.Alignment.CENTER)
    
    sphere = fem.geo3d.HalfSphere(50*mm, (L, wga/2, wgb/2), (1,0,0))

    objects = fem.geo3d.subtract(box1, box2)

    model.physics.set_frequency(15e9)

    model.geo.submit_domains(objects + [sphere])

    model.generate_mesh()

    gmsh.fltk.run()
