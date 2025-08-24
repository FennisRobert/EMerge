import emerge as em
import numpy as np
from emerge.plot import smith, plot_sp

""" STEPPED IMPEDANCE FILTER

In this demo we will look at how we can construct a stepped impedance filter using the
PCB Layouter interface in EMerge.

"""
# First we will define some constants/variables/parameters for our simulation.
mm = 0.001
mil = 0.0254*mm

L0, L1, L2, L3 = 400, 660, 660, 660 # The lengths of the sections in mil's
W0, W1, W2, W3 = 50, 128, 8, 224 # The widths of the sections in mil's

th = 62 # The PCB Thickness
er = 2.2 # The Dielectric constant

Hair = 60

## Material definition

# We can define the material using the Material class. Just supply the dielectric properties and you are done!
pcbmat = em.Material(er=er, tand=0.00, color="#217627", opacity=0.2)

# We start by creating our simulation object.

m = em.Simulation('Demo1_SIF', loglevel='INFO')
m.check_version("0.6.7") # Checks version compatibility.
# To accomodate PCB routing we make use of the PCBLayouter class. To use it we need to 
# supply it with a thickness, the desired air-box height, the units at which we supply
# the dimensions and the PCB material.

layouter = em.geo.PCB(th, unit=mil, material=pcbmat, layers=3)

# We will route our PCB using the "method chaining" syntax. First we call the .new() method
# to start a new trace. This will returna StripPath object on which we may call methods that
# sequentially constructs our stripline trace. In this case, it is sipmly a sequence of straight
# sections.

layouter.new(0,0,W0, (1,0), z=layouter.z(2)).store('p1').straight(L0, W0).straight(L1,W1).straight(L2,W2).straight(L3,W3)\
    .straight(L2,W2).straight(L1,W1).straight(L0,W0).store('p2')

# Next we generate a wave port surface to use for our simulation. A wave port can be automatically
# generated for a given stripoline section. To easily reference it we use the .ref() method to 
# recall the sections we created earlier.
p1 = layouter.modal_port(layouter.load('p1'), height=0)
p2 = layouter.modal_port(layouter.load('p2'), height=0)

# Finally we compile the stirpline into a polygon. The compile_paths function will return
# GeoSurface objects that form the polygon. Additionally, we may turn on the Merge feature
# which will then return a single GeoSurface type object that we can use later.
polies = layouter.compile_paths(True)

# We can manually define blocks for the dielectric or air or let the PCBLayouter do it for us.
# First we must determine the bounds of our PCB. This function by default will make a PCB
# just large enough to contain all the coordinates in it (in the XY plane). By adding extra
# margins we can make sure to add sufficient space next to the trace. Just make sure that there
# is no margin where the wave ports need to go.
layouter.determine_bounds(leftmargin=0, topmargin=200, rightmargin=0, bottommargin=200)

# We can now generate the PCB and air box. The material assignment is automatic!

pcb = layouter.generate_pcb(True, merge=True)

# We now pass all the geometries we have created to the .commit_geometry() method.
m.commit_geometry()

# We set our desired resolution (fraction of the wavelength)
m.mw.set_resolution(0.15)

# And we define our frequency range
m.mw.set_frequency_range(0.2e9, 8e9, 41)

# EMerge also has a convenient interface to improve surface meshing quality. 
# With the set_boundary_size(method) we can define a meshing resolution for the edges of boundaries.
# This is adviced for small stripline structures.
# The growth_rate setting allows us to change how fast the mesh size will recover to the original size.
m.mesher.set_boundary_size(polies, 2*mm, growth_rate=1.2)
m.mesher.set_face_size(p1, 2*mm)
m.mesher.set_face_size(p2, 2*mm)

# Finally we generate our mesh and view it
m.generate_mesh()

m.view()

# We can now define the modal ports for the in and outputs and set the conductor to PEC.
port1 = m.mw.bc.ModalPort(p1, 1, TEM=True)
port2 = m.mw.bc.ModalPort(p2, 2, TEM=True)
pec = m.mw.bc.PEC(polies)

## OPTIONAL
# If we want to view the port mode we have to first know it. The modes are computed using a modal analysis.
# Since the latest version of EMerge. Modal analysis is executed automatically. Here we compute them manually
# as the mode is only calculated during the frequency domain solution.
if False:
    # Make sure to set the TEM property to True so that
    # EMerge knows to handle the port mode as a TEM boundary. This also includes the automatic
    # determination of a voltage integration line used for computing the port impedance.
    m.mw.modal_analysis(port1, 1, TEM=True)
    m.mw.modal_analysis(port2, 1, TEM=True)

    # Finally we import the display class to view the resultant modes
    m.display.add_object(pcb, opacity=0.1)
    m.display.add_object(polies)
    m.display.add_portmode(port1, 21)
    m.display.add_portmode(port2, 21)
    m.display.show()

# Finally we execute the frequency domain sweep and compute the Scattering Parameters.
sol = m.mw.run_sweep(parallel=True, njobs=4, frequency_groups=8)

# Our "sol" variable is of type MWData (Microwave Data). This contains a set of scalar data 
# like S-parameters and field data like the E/H field. The scalar data is in sol.scalar and the 
# field data in sol.field. Our data is currently a large list of entries in the dataset simply in order
# at which it is computed. We can structure our data in cases where we do a single frequency sweep
# or with a structured parameter sweep. In this case we can use the .grid property which will attempt
# to construct an N-dimensional grid from our results. 

gritted_data = sol.scalar.grid

# The gritted_data is of type MWGlobalNdim which means its an N-dimensional set of data.
# From this we can simply take all that we need.

f = gritted_data.freq
S11 = gritted_data.S(1,1)
S21 = gritted_data.S(2,1)

# This extracts the actual simulation data.
plot_sp(f, [S11, S21], labels=['S11','S21'], dblim=[-40,6], logx=True)

# We can also supersample our data by constructing a model using the Vector Fitting algorithm

f = np.linspace(0.2e9, 8e9, 2001)
S11 = gritted_data.model_S(1,1,f)
S21 = gritted_data.model_S(2,1,f)

smith(S11, labels='S11', f=f)

plot_sp(f, [S11, S21], labels=['S11','S21'], dblim=[-40,6], logx=True)

m.display.add_object(pcb, opacity=0.1)
m.display.add_object(polies, opacity=0.5)
m.display.add_surf(*sol.field[0].cutplane(1*mm, z=-0.75*th*mil).scalar('Ez','real'))
m.display.show()