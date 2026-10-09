
def test():
    import emerge as em

    mm = 0.001
    a = 106*mm
    b = 30*mm
    H = 70*mm
    wga = 70*mm
    wgb = 18*mm
    fl = 25*mm

    # We start again by defining our simulation model
    model = em.Simulation('Periodic')

    periodic_cell = em.HexCell((-a/2, b/2, 0), (-a/2, -b/2, 0), (0, -b/2, 0))

    # To make sure that we can run a periodic simulation we must tell the simulation that
    # it has to copy the meshing on each face that is duplcated. We can simply pass our periodic
    # cell to our model using the set_periodic_cell() method.

    model.set_periodic_cell(periodic_cell)

    # We can easily use our periodic cell to construct volumes with the appropriate faces. We simply call the volume method
    # to construct a cell region from z=0 to z=H

    model['box'] = periodic_cell.volume(0, H)

    # We also create a waveguide foor the feed
    model['wg'] = em.geo.Box(wga,wgb,fl, (-wga/2, -wgb/2,-fl) )

    # Next we define our geometry as usual
    # Beause we stored our geometry in our model object using the get and set-item notation. We don't have to pass the items anymore.
    model.commit_geometry()

    model.mw.set_frequency_range(2.8e9, 3.3e9, 5)
    model.mw.set_resolution(0.1)

    # Then we create our mesh and view the result
    model.generate_mesh()

    # Now lets define our boundary conditions
    # First the waveguide port
    wg = model['wg']
    wgbc = model.mw.bc.RectangularWaveguide(wg.face('bottom'), 1)

    # And then the absorbing boundary at the top
    abc = model.mw.bc.AbsorbingBoundary(model['box'].face('back'))

    # We can use the set_scanangle method to set the appropriate phases for the boundary. The scan angle is defined as following
    # kx = sin(θ)·cos(ϕ)
    # ky = sin(θ)·sin(ϕ)
    # kx = cos(θ)ϕ
    # The arguments of the function are θ,ϕ in degrees.
    periodic_cell.set_scanangle(30,45)

    # And at last we run our simulation and view the results.
    model.set_solver(em.EMSolver.SUPERLU)
    data = model.mw.run_sweep()

    model.display.add_object(wg)
    model.display.add_object(model['box'])
    model.display.add_surf(*data.field[0].cutplane(3*mm, y=0).scalar('Ey','real'))
    model.display.add_surf(*data.field[0].cutplane(3*mm, x=0).scalar('Ey','real'))
    model.display._reset()

    em.cleanup()
if __name__=="__main__":
    test()