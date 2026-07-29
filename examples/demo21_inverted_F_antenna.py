import emerge as em
import numpy as np
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff

""" PCB INVERTED-F ANTENNA (IFA) DEMO

This design is modeled after Texas Instruments Application Note AN043
https://www.ti.com/lit/an/swra117d/swra117d.pdf

Developed by: Rasmus Luomaniemi
"""

# --- Unit and simulation parameters --------------------------------------
mm = 0.001  # meters per millimeter

# --- Antenna geometry dimensions ----------------------------------------
L1 = 3.94 * mm
L2 = 2.47 * mm
L3 = 4.76 * mm
L4 = 2.64 * mm
L5 = 1.77 * mm
L6 = 4.90 * mm
W1 = 0.90 * mm
W2 = 0.50 * mm
D1 = 0.50 * mm
D2 = 0.30 * mm
D3 = 0.30 * mm
D4 = 0.50 * mm
D5 = 0.65 * mm
D6 = 1.70 * mm

ground_W = D1 + L3 + L5 + L2 + L5 + L2 + D3
ground_L = 30 * mm

# Refined frequency range for antenna resonance around 2.4–2.5 GHz
f1 = 2e9  # start frequency
f2 = 3e9  # stop frequency

# --- Create simulation object -------------------------------------------
model = em.Simulation("IFA")
model.check_version("2.8.2")  # Checks version compatibility.

# --- Define geometry primitives -----------------------------------------
# Substrate block, thickness in Z (negative down)
dielectric = em.geo.Box(
    ground_W, ground_L + L6 - D4 + W2 + D2, 1 * mm, position=(-D1, -ground_L, -1 * mm)
)

# Metal antenna element sections on top of substrate
ant_1 = em.geo.Box(W1, L6 - D4, 0.035 * mm, position=(0, 0, 0))

ant_2 = em.geo.Box(L3, W2, 0.035 * mm, position=(0, L6 - D4, 0))

ant_3 = em.geo.Box(W2, L6 - D4 - 0.5 * mm, 0.035 * mm, position=(W1 + D5, 0.5 * mm, 0))

ant_4 = em.geo.Box(W2, L4, 0.035 * mm, position=(L3 - W2, L6 - D4 - L4, 0))

ant_5 = em.geo.Box(L5, W2, 0.035 * mm, position=(L3, L6 - D4 - L4, 0))

ant_6 = em.geo.Box(W2, L4, 0.035 * mm, position=(L3 + L5, L6 - D4 - L4, 0))

ant_7 = em.geo.Box(L2, W2, 0.035 * mm, position=(L3 + L5, L6 - D4, 0))

ant_8 = em.geo.Box(W2, L4, 0.035 * mm, position=(L3 + L5 + L2 - W2, L6 - D4 - L4, 0))

ant_9 = em.geo.Box(L5, W2, 0.035 * mm, position=(L3 + L5 + L2, L6 - D4 - L4, 0))

ant_10 = em.geo.Box(W2, L4, 0.035 * mm, position=(L3 + L5 + L2 + L5, L6 - D4 - L4, 0))

ant_11 = em.geo.Box(L2, W2, 0.035 * mm, position=(L3 + L5 + L2 + L5, L6 - D4, 0))

ant_12 = em.geo.Box(
    W2, L1, 0.035 * mm, position=(L3 + L5 + L2 + L5 + L2 - W2, L6 - D4 - L1, 0)
)

ant_element = em.geo.unite(
    ant_1,
    ant_2,
    ant_3,
    ant_4,
    ant_5,
    ant_6,
    ant_7,
    ant_8,
    ant_9,
    ant_10,
    ant_11,
    ant_12,
).set_material(em.lib.COPPER)

ground = em.geo.Box(
    ground_W, ground_L, 0.035 * mm, position=(-D1, -ground_L, 0)
).set_material(em.lib.COPPER)

# Air box
air = em.geo.open_region(15 * mm, 15 * mm, 15 * mm).background()
# Background makes sure no materials of overlapping domains are overwritten

# Plate defining lumped port geometry (origin + width/height vectors)
port = em.geo.Plate(
    np.array([W1 + D5, 0, 0.035 * mm]),  # lower port corner
    np.array([W2, 0, 0]),  # width vector along X
    np.array([0, 0.5 * mm, 0]),  # height vector along Y
)

# --- Assign materials and simulation settings ---------------------------
# Dielectric material with some transparency for display
dielectric.set_material(em.Material(4.0, color="#207020", opacity=0.9)).prio_down()

# Mesh resolution: fraction of wavelength
model.mw.set_resolution(0.33)

# Frequency sweep across the resonance
model.mw.set_frequency_range(f1, f2, 7)

# --- Combine geometry into simulation -----------------------------------
model.commit_geometry()

# --- Mesh refinement settings --------------------------------------------
# Refined mesh on port face for excitation accuracy
model.mesher.set_face_size(port, 0.25 * mm)
model.mesher.set_boundary_size(ant_element.face("-z"), 1 * mm)
# --- Generate mesh and preview ------------------------------------------
model.generate_mesh()  # build the finite-element mesh
model.view(selections=[port])  # show the mesh around the port

# --- Boundary conditions ------------------------------------------------
# Define lumped port with specified orientation and impedance
port_bc = model.mw.bc.LumpedPort(
    port, 1, width=W2, height=0.5 * mm, direction=em.YAX, Z0=50
)

# Predefining selection
# The outside of the air box for the absorbing boundary
boundary_selection = air.boundary()

# Assigning the boundary conditions
abc = model.mw.bc.AbsorbingBoundary(boundary_selection)

# --- Run frequency-domain solver ----------------------------------------
model.view(plot_mesh=True, volume_mesh=False)
model.view(bc=True)
data = model.mw.run_sweep()

# --- Post-process S-parameters ------------------------------------------
freqs = data.scalar.grid.freq
freq_dense = np.linspace(f1, f2, 1001)
S11 = data.scalar.grid.model_S(1, 1, freq_dense)  # reflection coefficient
plot_sp(freq_dense, S11)  # plot reflection coefficient in dB
smith(S11, f=freq_dense, labels="S11")  # Smith chart of S11

# --- Far-field radiation pattern ----------------------------------------
# Extract 2D cut at phi=0 plane and plot E-field magnitude
ff1 = data.field.find(freq=2.45e9).farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
ff2 = data.field.find(freq=2.45e9).farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)

plot_ff(
    ff1.ang * 180 / np.pi, [ff1.gain.norm, ff2.gain.norm], dB=True, ylabel="Gain [dBi]"
)  # linear plot vs theta
plot_ff_polar(
    ff1.ang, [ff1.gain.norm, ff2.gain.norm], dB=True, dBfloor=-20
)  # polar plot of radiation

# --- 3D radiation visualization -----------------------------------------
# Add geometry to 3D display
model.display.populate()
# Compute full 3D far-field and display surface colored by |E|q
field = data.field.find(freq=2.45e9)
ff3d = field.farfield_3d(boundary_selection, origin=(0, 0, 0))
model.display.add_farfield3d(ff3d, component="gain", dB=True, rmax=30 * mm, opacity=0.6)
model.display.show()
