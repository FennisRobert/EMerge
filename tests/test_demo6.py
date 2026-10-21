
def test():
    import emerge as em
    from emerge.plot import plot_sp

    """PCB Vias

    This demonstration shows how to add vias with the PCB router. Make sure to go through the other
    PCB related demos (demo1 and demo3) to get more information on the PCBLayouter.

    """

    mm = 0.001
    th = 1

    model = em.Simulation('Stripline_test', loglevel='DEBUG')
    # As usual we start by creating our layouter
    ly = em.geo.PCBNew(th, mm, em.GCS, layers=3, material=em.lib.DIEL_RO4350B, trace_material=em.lib.PEC)

    # Here we define a simple stripline path that makes a knick turn and a via jump to a new layer.
    # None of the transmission lines are conciously matched in any way, this is just about the routing

    # The .via(...) method allows one to add a via geometry to the PCBLayouter that can be extracted later
    # A user may decide whether to proceed or not beyond the via. More information can be found in the 
    # docstring of the method. The vias can be created later.
    w0 = ly.calc.z0(50, 1, 0)*0.9
    w1 = ly.calc.z0(50,2,0)
    ly.new(0,0,w0,(1,0), 1)['p1'].straight(10).turn(90)\
        .straight(10).turn(-90, corner_type='champher')\
        .straight(5).via(2, 0.2, True, width=w1)\
        .straight(8).via(1, 0.2, width=w0).straight(5)\
        .turn(-90).straight(10).turn(90).straight(10)['p2']
    # As usual we compile the traces as a merger of polygons
    trace = ly.compile_paths(True)

    # Now that we have via's defined, we can do the same with vias. I set Merge to True so that I get back
    # One GeoObject.
    vias = ly.generate_vias(True)

    # Here I use lumped ports instead of wave ports. I use the references made earlier to generate the port.
    # By default, all lumped port sheets will be shorted to z=-thickness. You can change this as an optional
    # argument.
    lp1 = ly.lumped_port(ly.load('p1'), 1)
    lp2 = ly.lumped_port(ly.load('p2'), 2)

    # Because lumped ports don't stop at the edge of our domain, we make sure to add some margins everywhere.
    ly.determine_bounds(5,5,5,5)

    # Finally we can generate the PCB volumes. Because the trace start halfway through the PCB we turn
    # on the split-z function which cuts the PCB in multiple layers. This improves meshing around the striplines.
    diel = ly.generate_pcb(True, merge=True)
    # We also define the air-box
    air = ly.generate_air(3)
    #model.view(use_gmsh=True)
    # The rest is as usual
    model.commit_geometry()

    model.mw.set_frequency_range(1e9, 6e9, 11)
    model.mesher.set_boundary_size(trace, 3*mm)

    model.generate_mesh()

    # We display the geometry with extra attention to the vias. With the vias.boundary() method we can
    # specifically show the outside faces of the via.

    # We setup the lumped port boundary conditions. Because of an added functionality in the PCBLayouter 
    # class, you don't have to specify the width, height and direction of the lumped port, this information
    # is contained in the lumped port sheet. You can see this information as its stored in the lp1._aux_data
    # dictionary.

    pec = model.mw.bc.PEC(trace)

    #We also add a PEC for the outsides of our via.
    pecvia = model.mw.bc.PEC(vias.boundary())

    # Finally we run the simulation!
    data = model.mw.run_sweep(True, 4, frequency_groups=8)

    freq = data.scalar.grid.freq
    S11 = data.scalar.grid.S(1,1)
    S21 = data.scalar.grid.S(2,1)

    plot_sp(freq/1e9, [S11, S21], labels=['S11','S21'], show_plot=False)

    model.display.add_object(diel, opacity=0.2)
    model.display.add_object(trace)
    model.display.add_object(vias)

    # In the latest version, you can use the cutplane method of the dataset class
    # which is equivalent to the interpolate method except it automatically generates
    # the point cloud based on a plane x,y or z coordinate.
    model.display.add_quiver(*data.field[3].cutplane(ds=0.001, z=-0.00025).vector('E'))
    model.display.add_surf(*data.field[3].cutplane(ds=0.001, z=-0.00075).scalar('Ez','real'))
    model.display._reset()
    
    em.cleanup()
    
if __name__=="__main__":
    test()