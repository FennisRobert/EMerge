import emerge as em
from emerge.plot import smith, plot_sp

""" STEPPED IMPEDANCE FILTER

In this demo we will look at how we can construct a stepped impedance filter using the
PCB Layouter interface in EMerge.

"""

# First we will define some constants/variables/parameters for our simulation.

mm = 0.001
mil = 0.0254 * mm

L0, L1, L2, L3 = 400, 660, 660, 660  # The lengths of the sections in mil's
W0, W1, W2, W3 = 50, 128, 8, 224  # The widths of the sections in mil's

th = 62  # The PCB Thickness
er = 2.2  # The Dielectric constant

Hair = 60

## Material definition

# We can define the material using the Material class. Just supply the dielectric properties and you are done!
pcbmat = em.Material(er=er, color="#217627", opacity=0.2)

# We start by creating our simulation object.

model = em.Simulation("SteppedImpedanceFilter")
model.check_version("2.8.8")  # Checks version compatibility.

# To accomodate PCB routing we make use of the PCBLayouter class. To use it we need to
# supply it with a thickness, the desired air-box height, the units at which we supply
# the dimensions and the PCB material.

layouter = em.geo.PCBNew(th, unit=mil, material=pcbmat, layers=3)

# We will route our PCB using the "method chaining" syntax. First we call the .new() method
# to start a new trace. This will returna StripPath object on which we may call methods that
# sequentially constructs our stripline trace. In this case, it is sipmly a sequence of straight
# sections.

layouter.new(0, 0, W0, (1, 0), z=layouter.z(1)).store("p1").straight(L0, W0).straight(
    L1, W1
).straight(L2, W2).straight(L3, W3).straight(L2, W2).straight(L1, W1).straight(
    L0, W0
).store("p2")

# Next we generate a wave port surface to use for our simulation. A wave port can be automatically
# generated for a given stripoline section. To easily reference it we use the .ref() method to
# recall the sections we created earlier.
p1 = layouter.modal_port(layouter.load("p1"), height=0)
p2 = layouter.modal_port(layouter.load("p2"), height=0)

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
model.commit_geometry()

# We set our desired resolution (fraction of the wavelength)
model.mw.set_resolution(0.25)

# And we define our frequency range
model.mw.set_frequency_range(0.2e9, 8e9, 41)

# EMerge also has a convenient interface to improve surface meshing quality.
# With the set_boundary_size(method) we can define a meshing resolution for the edges of boundaries.
# This is adviced for small stripline structures.
# The growth_rate setting allows us to change how fast the mesh size will recover to the original size.

model.mesher.set_boundary_size(polies, 1.5 * mm)
model.mesher.set_face_size(p1, 1 * mm)
model.mesher.set_face_size(p2, 1 * mm)

# Finally we generate our mesh and view it

model.generate_mesh()
model.view()


# We can now define the modal ports for the in and outputs and set the conductor to PEC.
port1 = model.mw.bc.ModalPort(p1, 1, modetype="TEM")
port2 = model.mw.bc.ModalPort(p2, 2, modetype="TEM")

# Finally we execute the frequency domain sweep and compute the Scattering Parameters.
sol = model.mw.run_sweep(parallel=True, n_workers=4, frequency_groups=8)

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
S11 = gritted_data.S(1, 1)
S21 = gritted_data.S(2, 1)
# This extracts the actual simulation data.
plot_sp(f, [S11, S21], labels=["S11", "S21"], dblim=[-40, 6], logx=True)

# We can also supersample our data by constructing a model using the Vector Fitting algorithm

# This will initialize a sub-sampled frequency axis

f = gritted_data.dense_f(2001)  # Dense frequencies are stored internally
S11 = gritted_data.model_S(
    1, 1
)  # The dense frequency points are available from the .dense_f() call.
S21 = gritted_data.model_S(2, 1)

smith(S11, labels="S11", f=f)

plot_sp(f, [S11, S21], labels=["S11", "S21"], dblim=[-40, 6], logx=True)

field = sol.field.find(freq=5.5e9)
model.display.add_object(pcb, opacity=0.1)
model.display.add_object(polies, opacity=0.5)
model.display.animate().add_field(
    field.cutplane(0.5 * mm, z=-0.75 * th * mil).scalar("Ez", "complex"),
    symmetrize=True,
)
model.display.add_portmode(port1, k0=field.k0)
model.display.add_portmode(port2, k0=field.k0)
model.display.show()
