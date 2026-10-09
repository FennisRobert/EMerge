def test():
    import emerge as em
    import numpy as np
    from emerge.plot import plot_ff_polar, plot_sp

    mm = 0.001
    wga = 22.86 * mm
    wgb = 10.16 * mm
    L = 50 * mm

    model = em.Simulation("Test Mode")

    # first lets define a WR90 waveguide
    wg_box = em.geo.Box(L, wga, wgb, position=(-L, -wga / 2, -wgb / 2))
    # Then define a capacitive iris cutout
    cutout = em.geo.Box(2 * mm, wga, wgb / 2, position=(-L / 2, -wga / 2, -wgb / 2))

    # remove the cutout from the box. Notice that we use a different name.
    # Geometry properties are not persistent after boolean operations so we
    # need the information of previous boxes.
    wg_box_new = em.geo.remove(wg_box, cutout)

    # define an air-box to radiat in.
    airbox = em.geo.Box(L / 2, L, L, position=(0, -L / 2, -L / 2))

    # Now define the geometry
    model.commit_geometry()

    # Lets define a frequency range for our simulation. This is needed
    # If we want to mesh our model.
    model.mw.set_frequency_range(8e9, 10e9, 7)

    # Now lets mesh our geometry
    model.generate_mesh()
    # model.view()
    ## We can now select faces and show them using the .view() interface

    # The box is defined in XYZ space. The sides left/right correspond to the
    # X-axis, The sides top/down to the Z-axis and front/back to the Y-axis.
    # We have to provide which original object we want to pick the left side from.
    feed_port = wg_box_new.face("left", tool=wg_box)

    # We can also select the outside and exclude a given face. Because our airbox
    # is not modified, we don't have to work with tools.
    radiation_boundary = airbox.boundary(exclude=("left",))

    radiation_boundary_2 = model.select.face.inlayer(1 * mm, 0, 0, (L, 0, 0))

    # Now lets define our simulation futher and do some farfield-computation!

    port = model.mw.bc.ModalPort(feed_port, 1)
    rad = model.mw.bc.AbsorbingBoundary(radiation_boundary)

    model.mw.modal_analysis(port, 1)

    # Run the simulation
    data = model.mw.run_sweep()

    # First the S11 plot
    f = data.scalar.grid.freq
    S11 = data.scalar.grid.S(1, 1)

    plot_sp(f / 1e9, S11, labels=["S11"], show_plot=False)

    # First we need to create a boundary mesh
    rad_surf = model.mesh.boundary_surface(radiation_boundary.tags, (0, 0, 0))
    # Then we need to compute the E-field on the edges.  We will pick the first frequency.
    Ein, Hin = data.field[0].interpolate(*rad_surf.exyz).EH

    em.cleanup()


if __name__ == "__main__":
    test()
