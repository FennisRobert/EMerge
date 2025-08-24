import emerge as em
import numpy as np
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff

""" PATCH ANTENNA DEMO

This design is modeled after this Comsol Demo: https://www.comsol.com/model/microstrip-patch-antenna-11742

In this demo we build and simulate a rectangular patch antenna on a dielectric
substrate with airbox and lumped port excitation, then visualize S-parameters
and far-field radiation patterns. """

# --- Unit and simulation parameters --------------------------------------
mm = 0.001              # meters per millimeter
margin = 5 * mm         # extra air margin around substrate
Nmodes = 1              # number of port modes to excite

# --- Antenna geometry dimensions ----------------------------------------
Wpatch = 53 * mm        # patch width (meters)
Lpatch = 52 * mm        # patch length
wline = 3.2 * mm        # feed line width
wstub = 7 * mm          # stub width for inset feed
lstub = 15.5 * mm       # stub (feed) length
wsub = 100 * mm         # substrate width
hsub = 100 * mm         # substrate length
th = 1.524 * mm         # substrate thickness

# --- Airbox and dielectric constants ------------------------------------
Hair = 40 * mm          # airbox height above substrate
er = 3.38               # substrate dielectric constant (relative)

# Refined frequency range for antenna resonance around 1.54–1.6 GHz
f1 = 1.54e9             # start frequency
f2 = 1.6e9              # stop frequency

# --- Create simulation object -------------------------------------------
# Using PVDisplay backend for 3D visualization
model = em.Simulation('MyPatchAntenna')
model.check_version("0.6.7") # Checks version compatibility.
# --- Define geometry primitives -----------------------------------------
# Substrate block centered at origin in XY, thickness in Z (negative down)
dielectric = em.geo.Box(wsub, hsub, th,
                        position=(-wsub/2, -hsub/2, -th))
# Air box above substrate (Z positive)
air = em.geo.Box(wsub, hsub, Hair,
                  position=(-wsub/2, -hsub/2, 0))

# Metal patch rectangle on top of substrate
rpatch = em.geo.XYPlate(Wpatch, Lpatch,
                        position=(-Wpatch/2, -Lpatch/2, 0))

# Define cutouts for inset feed: two rectangular plates to subtract
cutout1 = em.geo.XYPlate(wstub, lstub,
                         position=(-wline/2 - wstub, -Lpatch/2, 0))
cutout2 = em.geo.XYPlate(wstub, lstub,
                         position=( wline/2, -Lpatch/2, 0))
# Feed line plate to add back between cutouts
line = em.geo.XYPlate(wline, lstub,
                       position=(-wline/2, -Lpatch/2, 0))
# Plate defining lumped port geometry (origin + width/height vectors)
port = em.geo.Plate(
    np.array([-wline/2, -Lpatch/2, -th]),  # lower port corner
    np.array([wline, 0, 0]),                # width vector along X
    np.array([0, 0, th])                    # height vector along Z
)

# Build final patch shape: subtract cutouts, add feed line
rpatch = em.geo.remove(rpatch, cutout1)
rpatch = em.geo.remove(rpatch, cutout2)
rpatch = em.geo.add(rpatch, line)

# Assign copper material for visualization only
rpatch.material = em.lib.MET_COPPER

# --- Assign materials and simulation settings ---------------------------
# Dielectric material with some transparency for display
dielectric.material = em.Material(er, tand=0.01,
                                  color="#207020", opacity=0.6)
# Mesh resolution: fraction of wavelength
model.mw.resolution = 0.2
# Frequency sweep across the resonance
model.mw.set_frequency_range(f1, f2, 21)

# --- Combine geometry into simulation -----------------------------------
model.commit_geometry()

# --- Mesh refinement settings --------------------------------------------
# Finer boundary mesh on patch edges for accuracy
model.mesher.set_boundary_size(rpatch, 5 * mm, 1.1)
# Refined mesh on port face for excitation accuracy
model.mesher.set_face_size(port, 0.5 * mm)

# --- Generate mesh and preview ------------------------------------------
model.generate_mesh()                      # build the finite-element mesh
model.view(selections=[port])              # show the mesh around the port

# --- Boundary conditions ------------------------------------------------
# Define lumped port with specified orientation and impedance
port = model.mw.bc.LumpedPort(
    port, 1,
    width=wline, height=th,
    direction=em.ZAX,
    active=True, Z0=50
)
# Apply absorbing boundary on underside of airbox to simulate open space
boundary_selection = air.boundary(exclude=('bottom',))

model.view(selections=[boundary_selection,])
abc = model.mw.bc.AbsorbingBoundary(boundary_selection)
# Perfect conductor on the metallic patch
pec = model.mw.bc.PEC(rpatch)

# --- Run frequency-domain solver ----------------------------------------
data = model.mw.run_sweep()

# --- Post-process S-parameters ------------------------------------------
freqs = data.scalar.grid.freq
S11 = data.scalar.grid.S(1, 1)            # reflection coefficient
plot_sp(freqs, S11)                 # plot return loss in dB
smith(S11, f=freqs, labels='S11')         # Smith chart of S11

# --- Far-field radiation pattern ----------------------------------------
# Extract 2D cut at phi=0 plane and plot E-field magnitude
ff1 = data.field.find(freq=1.56e9)\
    .farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
ff2 = data.field.find(freq=1.56e9)\
    .farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)

plot_ff(ff1.ang*180/np.pi, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO])                # linear plot vs theta
plot_ff_polar(ff1.ang, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO])          # polar plot of radiation

# --- 3D radiation visualization -----------------------------------------
# Add geometry to 3D display
model.display.add_object(rpatch)
model.display.add_object(dielectric)
# Compute full 3D far-field and display surface colored by |E|
ff3d = data.field[1].farfield_3d(boundary_selection)
surf = ff3d.surfplot('normE', rmax=60 * mm,
                      offset=(0, 0, 20 * mm))
model.display.add_surf(*surf, cmap='viridis', symmetrize=False)
model.display.show()
