# -----------------------------------------------------------------------------
# Microstrip to slotline coupler and Adaptive mesh refinement
#
# In this example we will look at the setup of a microstripline to slotline coupler
# and see how we can use Adaptive mesh refinement in this simulation
# -----------------------------------------------------------------------------


import emerge as em
from emerge.plot import plot_sp


############################################################
#                    CONSTANT DEFINITION                   #
############################################################


mm = 0.001  # mm definition

th = 1.54  # PCB Thickness
Lprestub = 2  # Extra stripline before the radial stub
Lstub = 8  # Length of the radial stub
angstub = 100  # Angle of the radial stub
Lfeed = 20  # Length of the microstrip feedline
Lslot = 20  # Length of the slotline
wslot = 0.8  # width of the slotline
Lleft = 4  # Extra distance before the open-circuit circle
Rcirc = 8  # Readius of the open circuit circle.


############################################################
#                     SIMULATION SETUP                    #
############################################################

# We will invoke the SimulationBeta class because it houses some specific
# implementation details required for adaptive mesh refinement.
model = em.Simulation("Transition")
model.check_version("2.8.2")

# Next we create the PCB designer class instance.
pcb = em.geo.PCBNew(th, mm, material=em.lib.DIEL_RO4003C)

# We can use this function to compute the microstripline impedance. Right now (version 1.1) it just assumes top vs bottom layer.
w0 = pcb.calc.z0(50)

# We start a new simple microstripline at xy=(0,0) and go Lfeed forward.
pcb.new(0, 0, w0, (0, 1)).store("p1").straight(Lfeed).store("stop").straight(
    Lprestub
).store("stub")

# We load the StripLine element from the PCB where we stopped and use the coordinates to add a radial stub.
stub = pcb.load("stub")
pcb.radial_stub(stub.xy, Lstub, angstub, (0, 1), w0=w0)

# Then we use a botom layer stripline to create the slotline trace. We start a bit left of the stop point where
# and then proceed to the right.
stop = pcb.load("stop")
pcb.new(stop.x - Lleft, stop.y, wslot, (1, 0), pcb.z(0)).store("circ").straight(
    Lslot + Lleft
).store("p2")

# We load the XY coordinates of that last point
xydisc = pcb.load("circ").xy

# We create a circle geometry Such that the right side contacts the trace with 1mm overlap.
disc = em.geo.XYPolygon.circle(Rcirc * mm, 1 * mm).geo(
    em.GCS.displace((xydisc[0] - Rcirc + 1) * mm, xydisc[1] * mm, pcb.z(0) * mm)
)

# Then finally we generate all our paths.
# They will be returned in the following order
#    1. All stripline paths in order of creation
#    2. All polygon geometries

feed, slot, stub = pcb.compile_paths()

# After generation of all geometries we can determine the bounds. We will add 20mm of margin to the left and back of
# our PCB domain. We start at the front and end at the right.
pcb.determine_bounds(20, 20, 0, 0)

# We add a ground plane on the bottom.
ground = pcb.plane(pcb.z(0))

# We remove the slot line and circle from the ground plane
em.geo.subtract(ground, em.geo.unite(slot, disc))

# We add two modal port surfaces at the nodes p1 and p2
mp1 = pcb.modal_port(pcb.load("p1"), 5.0)
mp2 = pcb.modal_port(
    pcb.load("p2"), (5.0, 5.0), width_multiplier=15
)  # Wthe slot port width is 15 times the slot width.

# Finally we generate the PCB, top air and bottom air.
diel = pcb.generate_pcb()
air_top = pcb.generate_air(5)
air_bottom = pcb.generate_air(5, bottom=True)

# With Commit geometry we say: here we are done generating our model.
model.commit_geometry()

# We set the frequency range from 4GHz to 7GHz in 21 steps.
model.mw.set_frequency_range(4e9, 7e9, 21)

# We don't use any manual refinement steps to illustrate the power of
# Adaptive Mesh refinement.
# We generate the mesh and view it
model.mw.set_resolution(0.33)
model.generate_mesh()

# Notice the course initial mesh
model.view(plot_mesh=True)

############################################################
#                    BOUNDARY CONDITIONS                   #
############################################################

# Here we define our boundary conditions. Our first port is a (quasi-)TEM line.
p1 = model.mw.bc.ModalPort(mp1, 1)
# A pure slotline is also (quasi-)TEM but because its surrounded by a PEC boundary it
# Behaves like a TE mode. We set this explicitly to ensure proper extrapolation of the out-of-plane
# propagation constant
p2 = model.mw.bc.ModalPort(mp2, 2)
# We align the slotline mode along the positive Y-direction. This ensures that no polarity flips
# occur for the slotline mode that will add a random 180 degree phase shift in between
# refinement passes. The port phase is a degree of freedom that is not solved for during
# The modal port analysis so  this way we prevent that degree of freedom from showing up in
# S-parameter convergence study.

p2.align_modes(em.YAX)

# we also add an absorbing boundary at the top and bottom face to minimize resonance modes
model.mw.bc.AbsorbingBoundary(em.select(air_top.top, air_bottom.bottom))

############################################################
#                        SIMULATION                       #
############################################################

# First we call the Adaptive Mesh Refinement routine. We refine at 6GHz because this is in the pass-band.
# We set show_mesh to True so we can see the progress of refinement for the purspose of this example.
# This halts the simulation so we have to click away the window to proceed.
# You can see that more nodes are added around the signal traces because the E-field error is highest

# In this simulation setup, refinement does not converge due to the coarse starting point and the limited number of steps.
# This limitation is mostly set to make sure that users do not require a PC with large amounts of RAM to run this simulation.
# You may increase the number of steps at will to make the simulation converge.

model.adaptive_mesh_refinement(max_steps=8, frequency=5.5e9)

# We can view the improvement in the refined mesh.
model.view(plot_mesh=True, volume_mesh=False)

# Finally we start our solve with 4 parallel workers
data = model.mw.run_sweep(True, n_workers=4)


############################################################
#                      POST PROCESSING                     #
############################################################

# We make a convenient object name for the gridded S-parameter data and plot it.
g = data.scalar.grid

plot_sp(g.freq, [g.S(1, 1), g.S(2, 1)], labels=["S11", "S21"])

# Finally we make a 3D plot showing an animation of the Ez-field.
field = data.field.find(freq=6e9)
model.display.add_objects(*model.all_geos())
model.display.animate().add_field(
    field.cutplane(1 * mm, z=-th * mm / 2).scalar("Ez", "complex"), symmetrize=True
)
model.display.add_portmode(p1, k0=field.k0)
model.display.add_portmode(p2, k0=field.k0)
model.display.show()
