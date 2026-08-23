import emerge as em
import numpy as np
from emerge.plot import plot_sp, plot_ff_polar, smith

""" SLOTTED VIVALDI ANTENNA

In this demo we will look at the design of a slotted Vivaldi antenna in EMerge and how it can be setup.

The antenna is not perfect. The aim of this tutorial is to demonstrate how parametric curves are implemented in EMerge.
Additionally, the narrow regions are a perfect opportunity to demonstrate the efficiency of adaptive mesh refinement.

Due to the wind band nature of this simulation, it might take a while to simulate. 

The current version will consume up to 9GB of RAM with non SuperLU Solvers.

For ARM MacOS users it is reccommended to install the Accelerate solver using:
  - pip install git+https://github.com/FennisRobert/emerge-aasds

"""

model = em.Simulation("Vivaldi")
model.check_version("2.8.9")

mm = 0.001  # Millimeter
g = 0.3 * mm  # Narrow exponential taper slot gap size
L = 70 * mm  # Slot length
W = 55 * mm  # Aperturee width
K = 200  # Exponential curve growth factor
Radius = 20 * mm  # Open circuit circle radius
th = 0.5  # PCB Thickness
Lstub = 7  # Length of the radial stub
stub_ang = 80  # Radial stub angle
stub_ang_off = 20  # Radial stub angle offset
slot_margin = 5 * mm  # Margin around the taper for the slots
wslot = 4 * mm  # The width of the slot
wgap = 4 * mm  # The distance betwen slots

Wperiod = wslot + wgap


# Vivaldi curve parametric equations fy(t) and fx(t)
# t ∈ [0, 1]
# fx(t) = L*t
# fy(t) = (g/2) Kᵗ + (W-gK)/(2-2K) (1 - Kᵗ)

fx = lambda t: t * L
fy = lambda t: (g / 2) * K**t + (W - g * K) / (2 - 2 * K) * (1 - K**t)

# We will create the slots around the vivaldi by making slots that stretch across the entire
# PCB and then removing the exponential taper region dilated by some 5 millimeters.
# We could use a curve dilation but in this case we will derive the analytical expresion ourselves!

# This A-coeff is just to make the equations more brief.
A = (g / 2) - (W - g * K) / (2 - 2 * K)

# We need to find the derivative of dfy(t)/dfx
dfx = lambda t: A * np.log(K) / L * K ** (t)

# To create the dilated curve we have to formulate the parametric equation of a unit length tangent vector
# We know that the ratio of ty/tx is equal to the derivative of the curve so we can normalize the length
# by defining R(t) as
R = lambda t: 1 / np.sqrt(1 + dfx(t) ** 2)
# Such that
# t = R [f'(x), 1]

# Then the normal vector is simply
# n = R [1, -f'(x)]

# We add this scaled by the margin to our parametric equation to get:
fx2 = lambda t: t * L - slot_margin * R(t) * dfx(t)
fy2 = lambda t: fy(t) + slot_margin * R(t)

# Now that we have our parametric equations defined we first make an airbox and set its material priority to a low value.
# This makes sure that in overlapping regions, the air material will not be assigned. In this example it likely isn't necessary but it is good practice.
airbox = em.geo.Box(
    120 * mm, 80 * mm, 30 * mm, position=(-30 * mm, -40 * mm, -15 * mm)
).prio_set(1)

# Exponential taper
# We make our tapered region as following

# Step 1: We create an XYPolygon that is empty
# Step 2: We draw our first taper for the top curve. That is simply our parametric line
# Step 3: We make the return path being the bottom. This includes the same X-values and the Y-values inverted (-Y). And we must reverse the points from end to start.
# Step 4: Finally we turn our XYPolygon into a geometry by embedding it in a Coordinate System that is at the z-coordinate of the bottom of our PCB (-th).

exp_taper = (
    em.geo.XYPolygon()
    .parametric(fx, fy, tolerance=1e-5)
    .parametric(fx, lambda t: -fy(t), reverse=True, tolerance=1e-5)
    .geo(em.GCS.displace(0, 0, -th * mm))
)

# We do the same for the dilated one
exp_taper_dialated = (
    em.geo.XYPolygon()
    .parametric(fx2, fy2, tolerance=1e-5, tmax=2)
    .parametric(fx2, lambda t: -fy2(t), reverse=True, tolerance=1e-5, tmax=2)
    .geo(em.GCS.displace(0, 0, -th * mm))
)

# Next we generate 15 slots. We make them stretch across the entire PCB.
slots = []
for n in range(7):
    rect = em.geo.XYPlate(
        wslot, 60 * mm, (70 * mm - (n + 1) * Wperiod, -30 * mm, -th * mm)
    )
    slots.append(rect)

# Then we unite all the individual slots into one geometry.
slots = em.geo.unite(*slots)
# We subtract the dilated taper from our slots to only have the slots around the taper left.
slots = em.geo.subtract(slots, exp_taper_dialated)
# We also define the circular region that we use to offen an open circuit to the feed point.
disc = em.geo.Disc((-Radius / 2 + 1 * mm, 0, -th * mm), Radius / 2, (0, 0, 1))

# Our PCB is actually quite simple, it only contains the bottom Ground plane and upper feed trace.
pcbl = em.geo.PCBNew(th, mm, material=em.lib.DIEL_FR4)
# We compute the trace width for 50 ohms.
w0 = pcbl.calc.z0(50)
# Next we start a new trace at xy=(2mm, -10mm), we store this as our input port and move 10.5mm forwards just beyond where the taper slot is.
pcbl.new(2, -10, w0, (0, 1)).store("pin").straight(10.5).store("stub")

# we compute the direction in which our radial stub has to point. We rotate it away from the circle a bit.
dx = np.sin(stub_ang_off * np.pi / 180)
dy = np.cos(stub_ang_off * np.pi / 180)
rx, ry = pcbl.load(
    "stub"
).xy  # Here we take the X,Y coordinates at the end of our feed line

# Finally we place our stub. We lower is by 0.2mm to make it sit flush with the line due to the rotation.
pcbl.radial_stub((rx, ry - 0.2), Lstub, stub_ang, (dx, dy), w0=w0)

# Finallly we compile all the polygons into a single metal trace.
polies = pcbl.compile_paths(merge=True)

# We also create a rectangular region for our lumped port.
port = pcbl.lumped_port(pcbl.load("pin"))

# We manually define the X,Y bounds of the PCB because the vivaldi taper is not part of our PCB itself.
pcbl.set_bounds(xmin=-25, xmax=70, ymin=-30, ymax=30)
pcb = pcbl.generate_pcb()  # we generate the PCB delectricum
ground = pcbl.plane(-th)  # And ground

model.view(use_gmsh=True)

# We subtract the exponential taper, disc and slots from the ground plane.
ground = em.geo.subtract(ground, exp_taper)
ground = em.geo.subtract(ground, disc)
ground = em.geo.subtract(ground, slots)

# And we are done modelling!
model.commit_geometry()

# We set our frequency range from 3GHz to 10GHz in 21 setps.
model.mw.set_frequency_range(3e9, 8e9, 15)
model.mw.set_resolution(0.33)

# Here we set our boundary conditions. The Absorbing boundary surfaace is the outside of the airbox.
abc = airbox.outside()

# The lumped port is defined on the port rectangular region. All the dimensions are automatically stored inside
# the port geometry object when you create it with the .lumped_port() function so you don't have to pass them!
model.mw.bc.LumpedPort(port, 1, Z0=50)
model.mw.bc.AbsorbingBoundary(abc)
model.mesher.set_pec_face(polies)
model.mesher.set_pec_face(ground)

model.generate_mesh()

model.view(plot_mesh=True, volume_mesh=False)

# Before we run we call our adaptive mesh refinement at 7GHz. You can change the frequency yourself.
model.adaptive_mesh_refinement(frequency=7e9)
model.view(plot_mesh=True, volume_mesh=False)  # and view the resultant mesh

# Finally we start our sweep
data = model.mw.run_sweep(False)

# We extract the object that contains all our gritted global parameter simulation data such as the S-parameters
g = data.scalar.grid

# We create a denser grid of frequency points. This function takes the same value range as the simulation frequency but subsamples
# with a total of 1001 points.
fd = g.dense_f(1001)

# We can plot the S11 in the smith plot. We use model_S(1,1,fd) to interpolate our S-parameters using vector fitting
smith(g.model_S(1, 1, fd), f=fd)
S_non_reflected = np.sqrt(
    1 - np.abs(g.model_S(1, 1, fd)) ** 2
)  # The non reflected energy
plot_sp(fd, [g.model_S(1, 1, fd), S_non_reflected], labels=["S₁₁", "√(1-|S₁₁|²)"])

# We compute the farfield at 6Ghz
ff_data = data.field.find(freq=6e9).farfield_2d((1, 0, 0), (0, 0, 1), abc)
plot_ff_polar(
    ff_data.ang, ff_data.normE / em.EISO, dB=True, title="Farfield E-plane @ 6GHz"
)

# Finally we create a simple field plot.
ff_data = data.field.find(freq=6e9).farfield_3d(abc)
model.display.add_objects(*model.all_geos())
model.display.add_field(
    data.field[1].cutplane(0.8 * mm, z=-th * mm / 2).scalar("Ey", "real"),
    symmetrize=True,
)
model.display.add_farfield3d(
    ff_data, "normE", dB=True, dBfloor=-20, offset=(90 * mm, 0, 0), rmax=40 * mm
)
model.display.show()
